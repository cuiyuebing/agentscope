# -*- coding: utf-8 -*-
# pylint: disable=protected-access
"""Test cases for workspace pool mode — Docker and E2B backends.

Every test in this module exercises **real** sandbox infrastructure.  No mocks
are used.  The ``DockerWorkspace`` tests require a running Docker daemon; the
``E2BWorkspace`` tests additionally require the ``E2B_API_KEY`` environment
variable.

Test plan
---------

1. ``WorkspacePool`` (generic async pool):
     - Pre-warming fills up to ``min_idle``.
     - ``acquire`` returns a healthy workspace.
     - ``release`` recycles the workspace back into the pool.
     - ``max_reuse`` causes the entry to be destroyed and replenished.
     - ``stop`` drains and destroys all entries.

2. ``DockerWorkspaceManager`` in pool mode:
     - Context-manager start/stop lifecycle.
     - ``create_workspace`` checks out from the pool.
     - ``get_workspace`` returns the same active workspace.
     - ``close`` releases back to the pool.
     - Multiple sequential acquire-release cycles succeed.
     - Gateway health check works after resume.
     - Workspace isolation: state from one checkout does not leak to the next.

3. ``E2BWorkspaceManager`` in pool mode:
     - Same logical tests as Docker, but against E2B cloud sandboxes.

4. ``DockerWorkspace`` low-level pool lifecycle methods:
     - ``pause`` / ``resume`` round-trip.
     - ``light_reset_for_pool`` / ``heavy_reset_for_pool`` gateway restart.
     - ``gateway_health`` probe.
"""

import asyncio
import os
import shutil
import subprocess
import tempfile
import unittest
import uuid
from unittest.async_case import IsolatedAsyncioTestCase

from agentscope.app._manager._docker_workspace_manager import (
    DockerWorkspaceManager,
)
from agentscope.app._manager._e2b_workspace_manager import (
    E2BWorkspaceManager,
)
from agentscope.app._manager._workspace_pool import (
    PooledState,
    WorkspacePool,
)
from agentscope.workspace import DockerWorkspace, E2BWorkspace

# ── environment detection ─────────────────────────────────────────


def _docker_available() -> bool:
    """Return ``True`` iff the Docker daemon is reachable."""
    if shutil.which("docker") is None:
        return False
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            timeout=5,
            check=False,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return False


_DOCKER_OK = _docker_available()
_DOCKER_SKIP = "Docker daemon not available"

_E2B_API_KEY = os.getenv("E2B_API_KEY", "")
_E2B_SKIP = "E2B_API_KEY environment variable is not set"


# ═══════════════════════════════════════════════════════════════════
# Part 1: WorkspacePool with real Docker workspaces
# ═══════════════════════════════════════════════════════════════════


@unittest.skipUnless(_DOCKER_OK, _DOCKER_SKIP)
class TestWorkspacePoolWithDocker(IsolatedAsyncioTestCase):
    """Exercise ``WorkspacePool`` directly with real Docker containers.

    Uses small pool sizes (``min_idle=1``, ``max_idle=2``, ``total=3``)
    to keep test duration and resource usage manageable.
    """

    async def _factory(self) -> DockerWorkspace:
        """Create and initialize a fresh ephemeral DockerWorkspace."""
        ws = DockerWorkspace(
            workspace_id=None,
            workdir=None,  # ephemeral
        )
        await ws.initialize()
        return ws

    @staticmethod
    async def _reset(ws: DockerWorkspace) -> None:
        await ws.heavy_reset_for_pool()

    @staticmethod
    async def _health_check(ws: DockerWorkspace) -> bool:
        return await ws.gateway_health()

    @staticmethod
    async def _close(ws: DockerWorkspace) -> None:
        await ws.close()

    @staticmethod
    async def _pause(ws: DockerWorkspace) -> None:
        await ws.pause()

    @staticmethod
    async def _resume(ws: DockerWorkspace) -> None:
        await ws.resume()

    async def asyncSetUp(self) -> None:
        self.pool = WorkspacePool[DockerWorkspace](
            factory=self._factory,
            reset_fn=self._reset,
            health_check_fn=self._health_check,
            close_fn=self._close,
            pause_fn=self._pause,
            resume_fn=self._resume,
            min_idle=1,
            max_idle=2,
            total=3,
            create_batch_size=1,
            max_reuse=0,  # unlimited by default
            health_check_interval=600.0,  # disable periodic sweep
        )

    async def asyncTearDown(self) -> None:
        try:
            await self.pool.stop()
        except Exception:
            pass

    async def test_start_prewarms_to_min_idle(self) -> None:
        """``pool.start()`` pre-warms at least ``min_idle`` entries."""
        await self.pool.start()
        # After start, we should have at least 1 idle entry.
        self.assertGreaterEqual(self.pool.idle_count, 1)
        self.assertGreaterEqual(self.pool.total_managed, 1)

    async def test_acquire_returns_active_entry(self) -> None:
        """``acquire`` returns an entry in ACTIVE state with reuse_count=1."""
        await self.pool.start()
        entry = await self.pool.acquire()

        self.assertEqual(entry.state, PooledState.ACTIVE)
        self.assertEqual(entry.reuse_count, 1)
        self.assertIsNotNone(entry.workspace)
        self.assertTrue(entry.workspace.is_alive)

        # The workspace gateway should be healthy.
        healthy = await entry.workspace.gateway_health()
        self.assertTrue(healthy)

        # Clean up: release the entry back.
        await self.pool.release(entry)

    async def test_release_returns_entry_to_pool(self) -> None:
        """``release`` resets, health-checks, pauses, and re-pools the entry."""
        await self.pool.start()
        entry = await self.pool.acquire()
        ws_id = entry.workspace.workspace_id

        idle_before = self.pool.idle_count
        await self.pool.release(entry)

        # After release, idle count should increase.
        # Give some time for background replenishment to settle.
        await asyncio.sleep(0.5)
        self.assertGreaterEqual(self.pool.idle_count, idle_before)
        # The entry state should now be POOLED.
        self.assertEqual(entry.state, PooledState.POOLED)

    async def test_acquire_release_cycle(self) -> None:
        """Acquire-release-acquire cycle works: pool recycles workspaces."""
        await self.pool.start()

        # First cycle
        entry1 = await self.pool.acquire()
        self.assertTrue(entry1.workspace.is_alive)
        await self.pool.release(entry1)

        # Second cycle: should get a workspace from the pool
        entry2 = await self.pool.acquire()
        self.assertTrue(entry2.workspace.is_alive)
        healthy = await entry2.workspace.gateway_health()
        self.assertTrue(healthy)
        await self.pool.release(entry2)

    async def test_max_reuse_causes_destruction(self) -> None:
        """When ``max_reuse`` is reached, the entry is destroyed on release."""
        # Create a pool with max_reuse=1
        pool = WorkspacePool[DockerWorkspace](
            factory=self._factory,
            reset_fn=self._reset,
            health_check_fn=self._health_check,
            close_fn=self._close,
            pause_fn=self._pause,
            resume_fn=self._resume,
            min_idle=1,
            max_idle=1,
            total=2,
            create_batch_size=1,
            max_reuse=1,
            health_check_interval=600.0,
        )
        try:
            await pool.start()

            entry = await pool.acquire()
            self.assertEqual(entry.reuse_count, 1)
            ws_id_first = entry.workspace.workspace_id

            # Release should destroy since reuse_count (1) >= max_reuse (1).
            await pool.release(entry)
            self.assertEqual(entry.state, PooledState.DESTROYED)

            # Pool should replenish; acquire gives a *different* workspace.
            # Wait for replenishment.
            await asyncio.sleep(2)
            entry2 = await asyncio.wait_for(pool.acquire(), timeout=120)
            self.assertNotEqual(entry2.workspace.workspace_id, ws_id_first)
            await pool.release(entry2)
        finally:
            await pool.stop()

    async def test_stop_destroys_all(self) -> None:
        """``pool.stop()`` destroys all tracked entries."""
        await self.pool.start()

        entry = await self.pool.acquire()
        await self.pool.release(entry)

        total_before = self.pool.total_managed
        self.assertGreater(total_before, 0)

        await self.pool.stop()

        self.assertEqual(self.pool.total_managed, 0)
        self.assertEqual(self.pool.idle_count, 0)


# ═══════════════════════════════════════════════════════════════════
# Part 2: DockerWorkspace low-level pool lifecycle methods
# ═══════════════════════════════════════════════════════════════════


@unittest.skipUnless(_DOCKER_OK, _DOCKER_SKIP)
class TestDockerWorkspacePoolLifecycle(IsolatedAsyncioTestCase):
    """Test ``pause``, ``resume``, ``light_reset_for_pool``, and
    ``heavy_reset_for_pool`` on a real Docker container.
    """

    async def asyncSetUp(self) -> None:
        self.workspace = DockerWorkspace(
            workspace_id=f"pool-test-{uuid.uuid4().hex[:8]}",
            workdir=None,  # ephemeral
        )
        await self.workspace.initialize()

    async def asyncTearDown(self) -> None:
        try:
            await self.workspace.close()
        except Exception:
            pass

    async def test_gateway_health(self) -> None:
        """``gateway_health`` returns True on a live workspace."""
        self.assertTrue(await self.workspace.gateway_health())

    async def test_pause_resume_roundtrip(self) -> None:
        """``pause`` freezes the container; ``resume`` brings it back."""
        # Pre-condition: alive and healthy.
        self.assertTrue(self.workspace.is_alive)
        self.assertTrue(await self.workspace.gateway_health())

        # Pause
        await self.workspace.pause()
        self.assertFalse(self.workspace.is_alive)
        # Gateway should not be reachable while paused.
        self.assertFalse(await self.workspace.gateway_health())

        # Resume
        await self.workspace.resume()
        self.assertTrue(self.workspace.is_alive)
        self.assertTrue(await self.workspace.gateway_health())

    async def test_light_reset_for_pool(self) -> None:
        """``light_reset_for_pool`` restarts the gateway and wipes data.

        The container stays running; a new gateway token is minted.
        """
        old_token = self.workspace._gateway_token

        # Create some state inside the container.
        from agentscope.message import UserMsg

        await self.workspace.offload_context(
            "test_session",
            [UserMsg(name="user", content="before reset")],
        )

        await self.workspace.light_reset_for_pool()

        # Token should have changed.
        self.assertNotEqual(self.workspace._gateway_token, old_token)
        # Gateway should be healthy with new token.
        self.assertTrue(await self.workspace.gateway_health())
        # Sessions should be wiped.
        result = await self.workspace._exec(
            "ls /workspace/sessions/ 2>/dev/null",
        )
        content = result.stdout.decode().strip()
        self.assertEqual(content, "")

    async def test_heavy_reset_for_pool(self) -> None:
        """``heavy_reset_for_pool`` destroys and recreates the container.

        A completely fresh container is started; the workspace remains
        usable with a new gateway.
        """
        old_token = self.workspace._gateway_token

        # Create some state.
        from agentscope.message import UserMsg

        await self.workspace.offload_context(
            "test_session",
            [UserMsg(name="user", content="before heavy reset")],
        )

        await self.workspace.heavy_reset_for_pool()

        # Token should have changed.
        self.assertNotEqual(self.workspace._gateway_token, old_token)
        # Workspace should be alive and healthy.
        self.assertTrue(self.workspace.is_alive)
        self.assertTrue(await self.workspace.gateway_health())
        # Sessions dir should be empty (fresh container).
        result = await self.workspace._exec(
            "ls /workspace/sessions/ 2>/dev/null",
        )
        content = result.stdout.decode().strip()
        self.assertEqual(content, "")

    async def test_multiple_pause_resume_cycles(self) -> None:
        """Multiple pause/resume cycles remain stable."""
        for _ in range(3):
            await self.workspace.pause()
            self.assertFalse(self.workspace.is_alive)
            await self.workspace.resume()
            self.assertTrue(self.workspace.is_alive)
            self.assertTrue(await self.workspace.gateway_health())


# ═══════════════════════════════════════════════════════════════════
# Part 3: DockerWorkspaceManager pool mode
# ═══════════════════════════════════════════════════════════════════


@unittest.skipUnless(_DOCKER_OK, _DOCKER_SKIP)
class TestDockerWorkspaceManagerPoolMode(IsolatedAsyncioTestCase):
    """Test ``DockerWorkspaceManager`` with ``pool_enabled=True``.

    Verifies the full manager lifecycle: context-manager entry starts
    the pool, workspaces are checked out and returned, and exit
    destroys everything.
    """

    async def test_context_manager_lifecycle(self) -> None:
        """Entering the manager starts the pool; exiting stops it."""
        mgr = DockerWorkspaceManager(
            pool_enabled=True,
            min_idle=1,
            max_idle=1,
            total=2,
            create_batch_size=1,
        )
        async with mgr:
            self.assertIsNotNone(mgr._pool)
            self.assertGreaterEqual(mgr._pool.total_managed, 1)

        # After exit, pool should be drained.
        self.assertEqual(mgr._pool.total_managed, 0)

    async def test_create_workspace_returns_live_workspace(self) -> None:
        """``create_workspace`` checks out a live, healthy workspace."""
        mgr = DockerWorkspaceManager(
            pool_enabled=True,
            min_idle=1,
            max_idle=1,
            total=2,
            create_batch_size=1,
        )
        async with mgr:
            ws = await mgr.create_workspace("user1", "agent1", "sess1")
            self.assertIsInstance(ws, DockerWorkspace)
            self.assertTrue(ws.is_alive)
            self.assertTrue(await ws.gateway_health())
            # close returns it to pool
            await mgr.close(ws.workspace_id)

    async def test_get_workspace_returns_same_active(self) -> None:
        """``get_workspace`` returns the same workspace for a checked-out id."""
        mgr = DockerWorkspaceManager(
            pool_enabled=True,
            min_idle=1,
            max_idle=1,
            total=2,
            create_batch_size=1,
        )
        async with mgr:
            ws1 = await mgr.create_workspace("user1", "agent1", "sess1")
            wid = ws1.workspace_id
            # get_workspace with a key should return the same entry.
            # First, find the key used in _active.
            active_key = list(mgr._active.keys())[0]
            ws2 = await mgr.get_workspace(
                "user1",
                "agent1",
                "sess1",
                active_key,
            )
            self.assertIs(ws1, ws2)
            await mgr.close(active_key)

    async def test_close_releases_to_pool(self) -> None:
        """``close`` releases the workspace back to the pool."""
        mgr = DockerWorkspaceManager(
            pool_enabled=True,
            min_idle=1,
            max_idle=1,
            total=2,
            create_batch_size=1,
        )
        async with mgr:
            ws = await mgr.create_workspace("user1", "agent1", "sess1")
            active_key = list(mgr._active.keys())[0]
            self.assertIn(active_key, mgr._active)

            await mgr.close(active_key)
            # No longer in _active
            self.assertNotIn(active_key, mgr._active)

    async def test_sequential_create_close_cycles(self) -> None:
        """Multiple sequential create-close cycles succeed."""
        mgr = DockerWorkspaceManager(
            pool_enabled=True,
            min_idle=1,
            max_idle=1,
            total=2,
            create_batch_size=1,
        )
        async with mgr:
            for i in range(3):
                ws = await mgr.create_workspace(
                    f"user{i}",
                    f"agent{i}",
                    f"sess{i}",
                )
                self.assertTrue(ws.is_alive)
                self.assertTrue(await ws.gateway_health())
                active_key = list(mgr._active.keys())[0]
                await mgr.close(active_key)

    async def test_workspace_isolation_across_cycles(self) -> None:
        """State from one checkout does not leak to the next.

        Writes a file inside the workspace on the first checkout,
        releases it, then acquires again and verifies the file is gone
        (heavy reset destroys the entire container).
        """
        mgr = DockerWorkspaceManager(
            pool_enabled=True,
            min_idle=1,
            max_idle=1,
            total=2,
            create_batch_size=1,
        )
        async with mgr:
            # First checkout: write a marker file.
            ws1 = await mgr.create_workspace("user1", "agent1", "sess1")
            result = await ws1._exec(
                "echo 'leaked' > /workspace/leak_marker.txt",
            )
            self.assertTrue(result.ok())
            active_key = list(mgr._active.keys())[0]
            await mgr.close(active_key)

            # Second checkout: marker should not exist.
            ws2 = await mgr.create_workspace("user2", "agent2", "sess2")
            check = await ws2._exec(
                "cat /workspace/leak_marker.txt 2>/dev/null",
            )
            # The heavy_reset_for_pool creates a fresh container,
            # so the file should not exist.
            self.assertNotIn(
                "leaked",
                check.stdout.decode(),
            )
            active_key = list(mgr._active.keys())[0]
            await mgr.close(active_key)

    async def test_close_all(self) -> None:
        """``close_all`` releases every active workspace."""
        mgr = DockerWorkspaceManager(
            pool_enabled=True,
            min_idle=1,
            max_idle=2,
            total=3,
            create_batch_size=1,
        )
        async with mgr:
            ws1 = await mgr.create_workspace("u1", "a1", "s1")
            ws2 = await mgr.create_workspace("u2", "a2", "s2")
            self.assertEqual(len(mgr._active), 2)

            await mgr.close_all()
            self.assertEqual(len(mgr._active), 0)


# ═══════════════════════════════════════════════════════════════════
# Part 4: E2BWorkspace low-level pool lifecycle methods
# ═══════════════════════════════════════════════════════════════════


@unittest.skipUnless(_E2B_API_KEY, _E2B_SKIP)
class TestE2BWorkspacePoolLifecycle(IsolatedAsyncioTestCase):
    """Test ``pause``, ``resume``, ``light_reset_for_pool``, and
    ``heavy_reset_for_pool`` on a real E2B cloud sandbox.
    """

    async def asyncSetUp(self) -> None:
        self.workspace = E2BWorkspace(
            api_key=_E2B_API_KEY,
        )
        await self.workspace.initialize()

    async def asyncTearDown(self) -> None:
        try:
            # Kill rather than pause to free resources after test.
            if self.workspace._sandbox is not None:
                try:
                    await self.workspace._sandbox.kill()
                except Exception:
                    pass
                self.workspace._sandbox = None
            self.workspace.is_alive = False
        except Exception:
            pass

    async def test_gateway_health(self) -> None:
        """``gateway_health`` returns True on a live E2B workspace."""
        self.assertTrue(await self.workspace.gateway_health())

    async def test_pause_resume_roundtrip(self) -> None:
        """``pause`` pauses the sandbox; ``resume`` brings it back."""
        self.assertTrue(self.workspace.is_alive)
        self.assertTrue(await self.workspace.gateway_health())

        await self.workspace.pause()
        self.assertFalse(self.workspace.is_alive)

        await self.workspace.resume()
        self.assertTrue(self.workspace.is_alive)
        self.assertTrue(await self.workspace.gateway_health())

    async def test_light_reset_for_pool(self) -> None:
        """``light_reset_for_pool`` restarts gateway and wipes data."""
        old_token = self.workspace._gateway_token

        from agentscope.message import UserMsg

        await self.workspace.offload_context(
            "test_session",
            [UserMsg(name="user", content="before reset")],
        )

        await self.workspace.light_reset_for_pool()

        self.assertNotEqual(self.workspace._gateway_token, old_token)
        self.assertTrue(await self.workspace.gateway_health())

        # Sessions should be wiped.
        result = await self.workspace._exec(
            "ls /home/user/workspace/sessions/ 2>/dev/null",
        )
        content = result.stdout.decode().strip()
        self.assertEqual(content, "")

    async def test_heavy_reset_for_pool(self) -> None:
        """``heavy_reset_for_pool`` destroys and recreates the sandbox."""
        old_sandbox_id = self.workspace.sandbox_id
        old_token = self.workspace._gateway_token

        await self.workspace.heavy_reset_for_pool()

        # Sandbox should be different.
        self.assertNotEqual(self.workspace.sandbox_id, old_sandbox_id)
        self.assertNotEqual(self.workspace._gateway_token, old_token)
        self.assertTrue(self.workspace.is_alive)
        self.assertTrue(await self.workspace.gateway_health())


# ═══════════════════════════════════════════════════════════════════
# Part 5: E2BWorkspaceManager pool mode
# ═══════════════════════════════════════════════════════════════════


@unittest.skipUnless(_E2B_API_KEY, _E2B_SKIP)
class TestE2BWorkspaceManagerPoolMode(IsolatedAsyncioTestCase):
    """Test ``E2BWorkspaceManager`` with ``pool_enabled=True``.

    Uses small pool sizes to keep E2B billing manageable during tests.
    """

    async def test_context_manager_lifecycle(self) -> None:
        """Entering the manager starts the pool; exiting stops it."""
        mgr = E2BWorkspaceManager(
            api_key=_E2B_API_KEY,
            pool_enabled=True,
            min_idle=1,
            max_idle=1,
            total=2,
            create_batch_size=1,
        )
        async with mgr:
            self.assertIsNotNone(mgr._pool)
            self.assertGreaterEqual(mgr._pool.total_managed, 1)

        self.assertEqual(mgr._pool.total_managed, 0)

    async def test_create_workspace_returns_live_workspace(self) -> None:
        """``create_workspace`` checks out a live, healthy workspace."""
        mgr = E2BWorkspaceManager(
            api_key=_E2B_API_KEY,
            pool_enabled=True,
            min_idle=1,
            max_idle=1,
            total=2,
            create_batch_size=1,
        )
        async with mgr:
            ws = await mgr.create_workspace("user1", "agent1", "sess1")
            self.assertIsInstance(ws, E2BWorkspace)
            self.assertTrue(ws.is_alive)
            self.assertTrue(await ws.gateway_health())
            active_key = list(mgr._active.keys())[0]
            await mgr.close(active_key)

    async def test_get_workspace_returns_same_active(self) -> None:
        """``get_workspace`` returns the same workspace for a checked-out id."""
        mgr = E2BWorkspaceManager(
            api_key=_E2B_API_KEY,
            pool_enabled=True,
            min_idle=1,
            max_idle=1,
            total=2,
            create_batch_size=1,
        )
        async with mgr:
            ws1 = await mgr.create_workspace("user1", "agent1", "sess1")
            active_key = list(mgr._active.keys())[0]
            ws2 = await mgr.get_workspace(
                "user1",
                "agent1",
                "sess1",
                active_key,
            )
            self.assertIs(ws1, ws2)
            await mgr.close(active_key)

    async def test_close_releases_to_pool(self) -> None:
        """``close`` releases the workspace back to the pool."""
        mgr = E2BWorkspaceManager(
            api_key=_E2B_API_KEY,
            pool_enabled=True,
            min_idle=1,
            max_idle=1,
            total=2,
            create_batch_size=1,
        )
        async with mgr:
            ws = await mgr.create_workspace("user1", "agent1", "sess1")
            active_key = list(mgr._active.keys())[0]
            self.assertIn(active_key, mgr._active)

            await mgr.close(active_key)
            self.assertNotIn(active_key, mgr._active)

    async def test_sequential_create_close_cycles(self) -> None:
        """Multiple sequential create-close cycles succeed."""
        mgr = E2BWorkspaceManager(
            api_key=_E2B_API_KEY,
            pool_enabled=True,
            min_idle=1,
            max_idle=1,
            total=2,
            create_batch_size=1,
        )
        async with mgr:
            for i in range(2):
                ws = await mgr.create_workspace(
                    f"user{i}",
                    f"agent{i}",
                    f"sess{i}",
                )
                self.assertTrue(ws.is_alive)
                self.assertTrue(await ws.gateway_health())
                active_key = list(mgr._active.keys())[0]
                await mgr.close(active_key)

    async def test_workspace_isolation_across_cycles(self) -> None:
        """State from one checkout does not leak to the next.

        Writes a file inside the sandbox, releases, then acquires
        again and verifies the file is gone.
        """
        mgr = E2BWorkspaceManager(
            api_key=_E2B_API_KEY,
            pool_enabled=True,
            min_idle=1,
            max_idle=1,
            total=2,
            create_batch_size=1,
        )
        async with mgr:
            # First checkout: write a marker.
            ws1 = await mgr.create_workspace("user1", "agent1", "sess1")
            result = await ws1._exec(
                "echo 'leaked' > /home/user/workspace/leak_marker.txt",
            )
            self.assertTrue(result.ok())
            active_key = list(mgr._active.keys())[0]
            await mgr.close(active_key)

            # Second checkout: marker should not exist.
            ws2 = await mgr.create_workspace("user2", "agent2", "sess2")
            check = await ws2._exec(
                "cat /home/user/workspace/leak_marker.txt 2>/dev/null",
            )
            self.assertNotIn("leaked", check.stdout.decode())
            active_key = list(mgr._active.keys())[0]
            await mgr.close(active_key)


# ═══════════════════════════════════════════════════════════════════
# Part 6: WorkspacePool with real E2B sandboxes
# ═══════════════════════════════════════════════════════════════════


@unittest.skipUnless(_E2B_API_KEY, _E2B_SKIP)
class TestWorkspacePoolWithE2B(IsolatedAsyncioTestCase):
    """Exercise ``WorkspacePool`` directly with real E2B sandboxes."""

    async def _factory(self) -> E2BWorkspace:
        ws = E2BWorkspace(api_key=_E2B_API_KEY)
        await ws.initialize()
        return ws

    @staticmethod
    async def _reset(ws: E2BWorkspace) -> None:
        await ws.heavy_reset_for_pool()

    @staticmethod
    async def _health_check(ws: E2BWorkspace) -> bool:
        return await ws.gateway_health()

    @staticmethod
    async def _close(ws: E2BWorkspace) -> None:
        # Kill instead of pause for test cleanup.
        if ws._sandbox is not None:
            try:
                await ws._sandbox.kill()
            except Exception:
                pass
            ws._sandbox = None
        ws.is_alive = False

    @staticmethod
    async def _pause(ws: E2BWorkspace) -> None:
        await ws.pause()

    @staticmethod
    async def _resume(ws: E2BWorkspace) -> None:
        await ws.resume()

    async def asyncSetUp(self) -> None:
        self.pool = WorkspacePool[E2BWorkspace](
            factory=self._factory,
            reset_fn=self._reset,
            health_check_fn=self._health_check,
            close_fn=self._close,
            pause_fn=self._pause,
            resume_fn=self._resume,
            min_idle=1,
            max_idle=1,
            total=2,
            create_batch_size=1,
            max_reuse=0,
            health_check_interval=600.0,
        )

    async def asyncTearDown(self) -> None:
        try:
            await self.pool.stop()
        except Exception:
            pass

    async def test_start_prewarms(self) -> None:
        """``pool.start()`` pre-warms to ``min_idle``."""
        await self.pool.start()
        self.assertGreaterEqual(self.pool.idle_count, 1)

    async def test_acquire_release_cycle(self) -> None:
        """Acquire-release round-trip works with real E2B sandboxes."""
        await self.pool.start()

        entry = await self.pool.acquire()
        self.assertEqual(entry.state, PooledState.ACTIVE)
        self.assertTrue(entry.workspace.is_alive)
        self.assertTrue(await entry.workspace.gateway_health())

        await self.pool.release(entry)
        self.assertEqual(entry.state, PooledState.POOLED)

    async def test_stop_destroys_all(self) -> None:
        """``pool.stop()`` destroys all managed entries."""
        await self.pool.start()
        entry = await self.pool.acquire()
        await self.pool.release(entry)

        self.assertGreater(self.pool.total_managed, 0)
        await self.pool.stop()
        self.assertEqual(self.pool.total_managed, 0)


# ═══════════════════════════════════════════════════════════════════
# Part 7: DockerWorkspaceManager pool mode — advanced scenarios
# ═══════════════════════════════════════════════════════════════════


@unittest.skipUnless(_DOCKER_OK, _DOCKER_SKIP)
class TestDockerPoolModeAdvanced(IsolatedAsyncioTestCase):
    """Advanced pool-mode scenarios for Docker workspaces."""

    async def test_concurrent_checkouts(self) -> None:
        """Multiple workspaces can be checked out concurrently."""
        mgr = DockerWorkspaceManager(
            pool_enabled=True,
            min_idle=2,
            max_idle=3,
            total=4,
            create_batch_size=2,
        )
        async with mgr:
            ws1 = await mgr.create_workspace("u1", "a1", "s1")
            ws2 = await mgr.create_workspace("u2", "a2", "s2")

            self.assertTrue(ws1.is_alive)
            self.assertTrue(ws2.is_alive)
            self.assertIsNot(ws1, ws2)
            self.assertNotEqual(
                ws1.workspace_id,
                ws2.workspace_id,
            )

            # Both should be healthy independently.
            h1 = await ws1.gateway_health()
            h2 = await ws2.gateway_health()
            self.assertTrue(h1)
            self.assertTrue(h2)

            await mgr.close_all()

    async def test_pool_respects_total_cap(self) -> None:
        """Pool does not create more instances than ``total``."""
        mgr = DockerWorkspaceManager(
            pool_enabled=True,
            min_idle=1,
            max_idle=1,
            total=2,
            create_batch_size=1,
        )
        async with mgr:
            ws1 = await mgr.create_workspace("u1", "a1", "s1")
            ws2 = await mgr.create_workspace("u2", "a2", "s2")

            # total_managed should not exceed total (2).
            self.assertLessEqual(mgr._pool.total_managed, 2)

            await mgr.close_all()

    async def test_workspace_exec_after_pool_checkout(self) -> None:
        """Workspace can execute commands after pool checkout."""
        mgr = DockerWorkspaceManager(
            pool_enabled=True,
            min_idle=1,
            max_idle=1,
            total=2,
            create_batch_size=1,
        )
        async with mgr:
            ws = await mgr.create_workspace("u1", "a1", "s1")

            # Execute a command inside the container.
            result = await ws._exec("echo 'hello from pool'")
            self.assertTrue(result.ok())
            self.assertIn("hello from pool", result.stdout.decode())

            # Write and read back a file.
            await ws._exec("echo 'test data' > /workspace/test.txt")
            read_result = await ws._exec("cat /workspace/test.txt")
            self.assertTrue(read_result.ok())
            self.assertIn("test data", read_result.stdout.decode())

            active_key = list(mgr._active.keys())[0]
            await mgr.close(active_key)

    async def test_offload_after_pool_checkout(self) -> None:
        """Offload operations work on a pool-checked-out workspace."""
        mgr = DockerWorkspaceManager(
            pool_enabled=True,
            min_idle=1,
            max_idle=1,
            total=2,
            create_batch_size=1,
        )
        async with mgr:
            from agentscope.message import UserMsg

            ws = await mgr.create_workspace("u1", "a1", "s1")

            # Offload context should succeed.
            path = await ws.offload_context(
                "test_session",
                [UserMsg(name="user", content="Hello from pool!")],
            )
            self.assertTrue(path.endswith("context.jsonl"))

            # Verify the file was written inside the container.
            result = await ws._exec(f"cat {path}")
            self.assertTrue(result.ok())
            self.assertIn("Hello from pool!", result.stdout.decode())

            active_key = list(mgr._active.keys())[0]
            await mgr.close(active_key)

    async def test_list_tools_mcps_after_pool_checkout(self) -> None:
        """``list_tools`` and ``list_mcps`` work after pool checkout."""
        mgr = DockerWorkspaceManager(
            pool_enabled=True,
            min_idle=1,
            max_idle=1,
            total=2,
            create_batch_size=1,
        )
        async with mgr:
            ws = await mgr.create_workspace("u1", "a1", "s1")

            tools = await ws.list_tools()
            self.assertIsInstance(tools, list)
            # Docker workspace has no built-in tools (all via MCP).
            self.assertEqual(len(tools), 0)

            mcps = await ws.list_mcps()
            self.assertIsInstance(mcps, list)

            active_key = list(mgr._active.keys())[0]
            await mgr.close(active_key)


if __name__ == "__main__":
    unittest.main()
