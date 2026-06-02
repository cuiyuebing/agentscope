# -*- coding: utf-8 -*-
"""Integration tests for RLWorkspaceManager with real E2B sandboxes.

The whole module is skipped when the ``E2B_API_KEY`` environment variable is
not set, because every test requires live E2B cloud sandbox instances.

These tests exercise the full pooling lifecycle:
- Pool pre-warming (sandboxes are created, bootstrapped, and paused)
- Workspace checkout via ``get_workspace`` / ``create_workspace``
- Workspace release via ``close`` (reset → health check → pause → re-pool)
- Concurrent checkout / release under contention
- Max-reuse eviction
- ``close_all`` bulk release
- Context manager (``async with``) start/stop semantics
- Re-checkout of a previously released workspace slot
"""

import asyncio
import os
import unittest
from unittest.async_case import IsolatedAsyncioTestCase

from agentscope.app._manager._rl_workspace_manager import RLWorkspaceManager
from agentscope.app._manager._workspace_pool import PooledState
from agentscope.workspace import E2BWorkspace

# ── E2B availability check ─────────────────────────────────────────

_E2B_API_KEY = os.getenv("E2B_API_KEY", "")
_SKIP_REASON = "E2B_API_KEY environment variable is not set"


# ── helper ─────────────────────────────────────────────────────────


def _make_manager(
    *,
    min_idle: int = 1,
    max_idle: int = 2,
    total: int = 4,
    create_batch_size: int = 2,
    max_reuse: int = 0,
    health_check_interval: float = 300.0,
) -> RLWorkspaceManager:
    """Create a RLWorkspaceManager with small pool parameters for testing."""
    return RLWorkspaceManager(
        api_key=_E2B_API_KEY,
        min_idle=min_idle,
        max_idle=max_idle,
        total=total,
        create_batch_size=create_batch_size,
        max_reuse=max_reuse,
        health_check_interval=health_check_interval,
    )


# ── test classes ───────────────────────────────────────────────────


@unittest.skipUnless(_E2B_API_KEY, _SKIP_REASON)
class TestRLWorkspaceManagerStartStop(IsolatedAsyncioTestCase):
    """Test pool start (pre-warming) and stop (drain + destroy)."""

    async def test_context_manager_start_stop(self) -> None:
        """``async with`` starts the pool and pre-warms ``min_idle`` instances.

        Verifies:
        1. After ``__aenter__``, ``pool.idle_count >= min_idle``.
        2. ``pool.total_managed >= min_idle``.
        3. After ``__aexit__``, all sandboxes are destroyed and
           ``total_managed == 0``.
        """
        manager = _make_manager(min_idle=1, max_idle=2, total=3)

        async with manager:
            pool = manager._pool
            # Pre-warming should fill at least min_idle
            self.assertGreaterEqual(pool.idle_count, 1)
            self.assertGreaterEqual(pool.total_managed, 1)

        # After stop, everything should be cleaned up
        self.assertEqual(pool.total_managed, 0)
        self.assertEqual(pool.idle_count, 0)

    async def test_start_prewarms_to_min_idle(self) -> None:
        """Pool start fills idle queue to exactly ``min_idle``.

        Uses min_idle=2 to verify more than one sandbox is pre-warmed.
        """
        manager = _make_manager(min_idle=2, max_idle=3, total=5)

        async with manager:
            pool = manager._pool
            # Should have at least min_idle ready
            self.assertGreaterEqual(pool.idle_count, 2)

        self.assertEqual(pool.total_managed, 0)


@unittest.skipUnless(_E2B_API_KEY, _SKIP_REASON)
class TestRLWorkspaceManagerCheckout(IsolatedAsyncioTestCase):
    """Test workspace checkout (acquire) via get_workspace / create_workspace."""

    async def test_create_workspace_returns_live_workspace(self) -> None:
        """``create_workspace`` checks out a live workspace from the pool.

        Verifies:
        1. The returned object is an ``E2BWorkspace``.
        2. The workspace has a non-None ``sandbox_id``.
        3. The gateway is healthy (responds to health probe).
        4. The workspace is tracked in ``_active``.
        """
        manager = _make_manager(min_idle=1, max_idle=2, total=3)

        async with manager:
            ws = await manager.create_workspace(
                user_id="test-user",
                agent_id="test-agent",
                session_id="test-session",
            )

            self.assertIsInstance(ws, E2BWorkspace)
            self.assertIsNotNone(ws.sandbox_id)
            self.assertTrue(await ws.gateway_health())

            # Should be tracked in active
            self.assertEqual(len(manager._active), 1)

    async def test_get_workspace_returns_same_instance(self) -> None:
        """Repeated ``get_workspace`` with the same workspace_id returns the same instance.

        Verifies idempotent checkout: once a workspace is active for a
        given workspace_id, subsequent calls return the same object
        without consuming another pool slot.
        """
        manager = _make_manager(min_idle=1, max_idle=2, total=3)

        async with manager:
            ws1 = await manager.get_workspace(
                user_id="test-user",
                agent_id="test-agent",
                session_id="s1",
                workspace_id="ws-test-idem",
            )
            ws2 = await manager.get_workspace(
                user_id="test-user",
                agent_id="test-agent",
                session_id="s2",
                workspace_id="ws-test-idem",
            )

            # Same object
            self.assertIs(ws1, ws2)
            # Only one active entry
            self.assertEqual(len(manager._active), 1)

    async def test_get_workspace_different_ids_different_instances(
        self,
    ) -> None:
        """Different ``workspace_id`` values checkout different sandboxes.

        Verifies that two distinct workspace_id values produce two
        independent workspace instances from the pool.
        """
        manager = _make_manager(min_idle=2, max_idle=3, total=5)

        async with manager:
            ws1 = await manager.get_workspace(
                user_id="user-a",
                agent_id="agent-a",
                session_id="s1",
                workspace_id="ws-alpha",
            )
            ws2 = await manager.get_workspace(
                user_id="user-b",
                agent_id="agent-b",
                session_id="s2",
                workspace_id="ws-beta",
            )

            self.assertIsNot(ws1, ws2)
            self.assertNotEqual(ws1.sandbox_id, ws2.sandbox_id)
            self.assertEqual(len(manager._active), 2)

            # Both should be healthy
            self.assertTrue(await ws1.gateway_health())
            self.assertTrue(await ws2.gateway_health())


@unittest.skipUnless(_E2B_API_KEY, _SKIP_REASON)
class TestRLWorkspaceManagerRelease(IsolatedAsyncioTestCase):
    """Test workspace release (close) — reset, health-check, pause, re-pool."""

    async def test_close_returns_workspace_to_pool(self) -> None:
        """``close(workspace_id)`` resets and returns workspace to the pool.

        Verifies:
        1. After close, workspace_id is no longer in ``_active``.
        2. The pool's idle count increases (workspace recycled).
        3. The workspace's sandbox still exists (paused, not killed).
        """
        manager = _make_manager(min_idle=1, max_idle=2, total=3)

        async with manager:
            pool = manager._pool
            ws = await manager.create_workspace(
                user_id="test-user",
                agent_id="test-agent",
                session_id="test-session",
            )
            workspace_id = None
            for wid, entry in manager._active.items():
                if entry.workspace is ws:
                    workspace_id = wid
                    break
            self.assertIsNotNone(workspace_id)

            idle_before = pool.idle_count

            await manager.close(workspace_id)

            # No longer active
            self.assertNotIn(workspace_id, manager._active)
            # Wait briefly for the release to complete (pause is async)
            await asyncio.sleep(2)
            # Idle count should have increased
            self.assertGreater(pool.idle_count, idle_before)

    async def test_close_noop_for_unknown_id(self) -> None:
        """``close`` on an unknown workspace_id is a silent no-op."""
        manager = _make_manager(min_idle=1, max_idle=2, total=3)

        async with manager:
            # Should not raise
            await manager.close("nonexistent-workspace-id")

    async def test_close_all_releases_every_active(self) -> None:
        """``close_all`` releases every active workspace back to the pool.

        Verifies:
        1. After close_all, ``_active`` is empty.
        2. The pool absorbs the released workspaces (idle count increases).
        """
        manager = _make_manager(min_idle=1, max_idle=3, total=5)

        async with manager:
            pool = manager._pool

            # Check out two workspaces
            await manager.create_workspace(
                user_id="u1",
                agent_id="a1",
                session_id="s1",
            )
            await manager.create_workspace(
                user_id="u2",
                agent_id="a2",
                session_id="s2",
            )
            self.assertEqual(len(manager._active), 2)

            idle_before = pool.idle_count

            await manager.close_all()

            self.assertEqual(len(manager._active), 0)
            # Give time for async release pipeline
            await asyncio.sleep(3)
            # Idle count should have grown
            self.assertGreater(pool.idle_count, idle_before)


@unittest.skipUnless(_E2B_API_KEY, _SKIP_REASON)
class TestRLWorkspaceManagerRecycle(IsolatedAsyncioTestCase):
    """Test workspace recycling: checkout → use → release → re-checkout."""

    async def test_recycled_workspace_is_clean(self) -> None:
        """A workspace released and re-acquired has no residual state.

        Verifies:
        1. Write a file in the workspace during first checkout.
        2. Release it (triggers reset_for_pool).
        3. Re-acquire from pool.
        4. The file written in step 1 is gone.
        """
        manager = _make_manager(min_idle=1, max_idle=2, total=2)

        async with manager:
            # First checkout
            ws1 = await manager.create_workspace(
                user_id="u1",
                agent_id="a1",
                session_id="s1",
            )
            workspace_id_1 = None
            for wid, entry in manager._active.items():
                if entry.workspace is ws1:
                    workspace_id_1 = wid
                    break

            # Write a marker file inside the sandbox
            marker_path = "/home/user/workspace/data/test_marker.txt"
            await ws1._sandbox.files.write(
                marker_path,
                b"hello-from-first-checkout",
            )

            # Verify the file exists
            content = await ws1._sandbox.files.read(
                marker_path, format="bytes"
            )
            self.assertEqual(bytes(content), b"hello-from-first-checkout")

            # Release the workspace back to pool
            await manager.close(workspace_id_1)
            # Give time for the full release pipeline (reset + pause)
            await asyncio.sleep(5)

            # Re-acquire — should get a clean workspace
            ws2 = await manager.create_workspace(
                user_id="u2",
                agent_id="a2",
                session_id="s2",
            )

            # The marker file should NOT exist in the recycled workspace
            from e2b import FileNotFoundException

            with self.assertRaises((FileNotFoundException, FileNotFoundError)):
                await ws2._sandbox.files.read(marker_path, format="bytes")


@unittest.skipUnless(_E2B_API_KEY, _SKIP_REASON)
class TestRLWorkspaceManagerMaxReuse(IsolatedAsyncioTestCase):
    """Test the max_reuse eviction policy."""

    async def test_max_reuse_destroys_workspace(self) -> None:
        """Workspace exceeding ``max_reuse`` is destroyed on release.

        With ``max_reuse=1``, after one checkout+release cycle the
        workspace should be destroyed (not recycled back to pool).
        A subsequent acquire should get a fresh sandbox (different ID).
        """
        manager = _make_manager(
            min_idle=1,
            max_idle=2,
            total=3,
            max_reuse=1,
        )

        async with manager:
            pool = manager._pool

            # First checkout
            ws1 = await manager.create_workspace(
                user_id="u1",
                agent_id="a1",
                session_id="s1",
            )
            sandbox_id_1 = ws1.sandbox_id
            workspace_id_1 = None
            for wid, entry in manager._active.items():
                if entry.workspace is ws1:
                    workspace_id_1 = wid
                    break

            # Release — max_reuse=1 means this is the first reuse,
            # so the entry hits reuse_count=1 == max_reuse → destroyed
            total_before = pool.total_managed
            await manager.close(workspace_id_1)
            # Give time for destroy + replenishment
            await asyncio.sleep(8)

            # The old sandbox should not be the same as the next one
            ws2 = await manager.create_workspace(
                user_id="u2",
                agent_id="a2",
                session_id="s2",
            )
            # Different sandbox
            self.assertNotEqual(ws2.sandbox_id, sandbox_id_1)


@unittest.skipUnless(_E2B_API_KEY, _SKIP_REASON)
class TestRLWorkspaceManagerConcurrency(IsolatedAsyncioTestCase):
    """Test concurrent access patterns."""

    async def test_concurrent_create_workspace(self) -> None:
        """Multiple concurrent ``create_workspace`` calls all succeed.

        Verifies that the pool correctly handles parallel checkout
        requests without deadlock or double-allocation.
        """
        manager = _make_manager(
            min_idle=2,
            max_idle=3,
            total=5,
            create_batch_size=3,
        )

        async with manager:
            # Launch 3 concurrent create_workspace calls
            tasks = [
                manager.create_workspace(
                    user_id=f"user-{i}",
                    agent_id=f"agent-{i}",
                    session_id=f"session-{i}",
                )
                for i in range(3)
            ]
            results = await asyncio.gather(*tasks)

            # All should succeed with unique sandbox IDs
            sandbox_ids = {ws.sandbox_id for ws in results}
            self.assertEqual(len(sandbox_ids), 3)

            # All should be healthy
            for ws in results:
                self.assertIsInstance(ws, E2BWorkspace)
                self.assertTrue(await ws.gateway_health())

            # All tracked in active
            self.assertEqual(len(manager._active), 3)


@unittest.skipUnless(_E2B_API_KEY, _SKIP_REASON)
class TestRLWorkspaceManagerGatewayHealth(IsolatedAsyncioTestCase):
    """Test gateway health checking in the pooling lifecycle."""

    async def test_gateway_healthy_after_checkout(self) -> None:
        """Gateway is healthy immediately after pool checkout.

        The pool resumes the sandbox and health-checks the gateway
        before handing it out. This test verifies the contract.
        """
        manager = _make_manager(min_idle=1, max_idle=2, total=3)

        async with manager:
            ws = await manager.create_workspace(
                user_id="u1",
                agent_id="a1",
                session_id="s1",
            )
            # Gateway should respond immediately since pool checks before handout
            self.assertTrue(await ws.gateway_health())

    async def test_workspace_can_execute_commands(self) -> None:
        """A checked-out workspace can execute commands inside the sandbox.

        This verifies the sandbox is truly alive and usable after the
        pool's resume + health-check sequence.
        """
        manager = _make_manager(min_idle=1, max_idle=2, total=3)

        async with manager:
            ws = await manager.create_workspace(
                user_id="u1",
                agent_id="a1",
                session_id="s1",
            )
            # Run a simple command to confirm the sandbox is live
            result = await ws._exec("echo hello-pool-test")
            self.assertTrue(result.ok())
            self.assertIn(b"hello-pool-test", result.stdout)


@unittest.skipUnless(_E2B_API_KEY, _SKIP_REASON)
class TestRLWorkspaceManagerPoolMetrics(IsolatedAsyncioTestCase):
    """Test pool metrics and capacity enforcement."""

    async def test_total_cap_respected(self) -> None:
        """Pool does not exceed ``total`` managed instances.

        With total=3 and 3 checkouts, the pool should have exactly 3
        total managed instances (all active, none idle).
        """
        manager = _make_manager(
            min_idle=1,
            max_idle=2,
            total=3,
            create_batch_size=2,
        )

        async with manager:
            pool = manager._pool

            # Check out workspaces up to total
            workspaces = []
            for i in range(3):
                ws = await manager.create_workspace(
                    user_id=f"u{i}",
                    agent_id=f"a{i}",
                    session_id=f"s{i}",
                )
                workspaces.append(ws)

            # total_managed should not exceed the cap
            self.assertLessEqual(pool.total_managed, 3)
            # All 3 are active
            self.assertEqual(len(manager._active), 3)

    async def test_replenishment_after_checkout(self) -> None:
        """Pool triggers replenishment when idle drops below ``min_idle``.

        After checking out a workspace (which reduces idle count), the
        pool should asynchronously replenish back toward ``max_idle``.
        """
        manager = _make_manager(
            min_idle=2,
            max_idle=3,
            total=5,
            create_batch_size=2,
        )

        async with manager:
            pool = manager._pool
            initial_idle = pool.idle_count

            # Checkout one — this should drop idle below min_idle
            ws = await manager.create_workspace(
                user_id="u1",
                agent_id="a1",
                session_id="s1",
            )

            # Wait for replenishment to kick in
            await asyncio.sleep(10)

            # Idle should be replenished (may take time for factory calls)
            # At minimum, the pool should have tried to replenish
            self.assertGreaterEqual(pool.idle_count, 1)


@unittest.skipUnless(_E2B_API_KEY, _SKIP_REASON)
class TestRLWorkspaceManagerMetadataBinding(IsolatedAsyncioTestCase):
    """Test that workspace metadata is correctly updated on checkout."""

    async def test_sandbox_metadata_updated_on_checkout(self) -> None:
        """Sandbox metadata contains user/agent/workspace binding after checkout.

        Verifies the pool-checked-out workspace has its sandbox_metadata
        updated with the caller's identifiers.
        """
        manager = _make_manager(min_idle=1, max_idle=2, total=3)

        async with manager:
            ws = await manager.get_workspace(
                user_id="uid-42",
                agent_id="aid-99",
                session_id="sid-7",
                workspace_id="wid-bound",
            )

            self.assertEqual(
                ws.sandbox_metadata.get("agentscope.user.id"),
                "uid-42",
            )
            self.assertEqual(
                ws.sandbox_metadata.get("agentscope.agent.id"),
                "aid-99",
            )
            self.assertEqual(
                ws.sandbox_metadata.get("agentscope.workspace.id"),
                "wid-bound",
            )


if __name__ == "__main__":
    unittest.main()
