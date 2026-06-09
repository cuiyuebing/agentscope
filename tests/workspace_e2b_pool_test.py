# -*- coding: utf-8 -*-
# pylint: disable=protected-access
"""Integration tests for :class:`E2BWorkspaceManager` pool mode.

Every test in this module creates real E2B cloud sandboxes via the
workspace manager's pool. The whole module is skipped when the
``E2B_API_KEY`` environment variable is not set.
"""

import os
import unittest
from unittest.async_case import IsolatedAsyncioTestCase

from agentscope.app.workspace_manager._e2b_workspace_manager import (
    E2BWorkspaceManager,
)
from agentscope.app.workspace_manager._workspace_pool import (
    PooledState,
)

# ── E2B availability check ────────────────────────────────────────

_E2B_API_KEY = os.getenv("E2B_API_KEY", "")
_SKIP_REASON = "E2B_API_KEY environment variable is not set"


# ── tests ──────────────────────────────────────────────────────────


@unittest.skipUnless(_E2B_API_KEY, _SKIP_REASON)
class TestE2BPoolLifecycle(IsolatedAsyncioTestCase):
    """Pool lifecycle: start, acquire, release, stop with real sandboxes."""

    async def asyncSetUp(self) -> None:
        self.mgr = E2BWorkspaceManager(
            api_key=_E2B_API_KEY,
            pool_enabled=True,
            pool_min_ready=1,
            pool_max_ready=2,
            pool_capacity=3,
            pool_batch_size=1,
        )
        await self.mgr.__aenter__()

    async def asyncTearDown(self) -> None:
        await self.mgr.__aexit__(None, None, None)

    async def test_create_and_close(self) -> None:
        """create_workspace returns a live workspace; close releases it."""
        ws = await self.mgr.create_workspace(
            user_id="u1",
            agent_id="a1",
            session_id="s1",
        )
        self.assertTrue(ws.is_alive)
        self.assertIsNotNone(ws.workspace_id)

        wid = list(self.mgr._active.keys())[0]
        await self.mgr.close(wid)
        self.assertNotIn(wid, self.mgr._active)

    async def test_get_workspace_reuses_active(self) -> None:
        """Repeated get_workspace with same id returns the same workspace."""
        ws1 = await self.mgr.create_workspace(
            user_id="u1",
            agent_id="a1",
            session_id="s1",
        )
        wid = list(self.mgr._active.keys())[0]

        ws2 = await self.mgr.get_workspace(
            user_id="u1",
            agent_id="a1",
            session_id="s2",
            workspace_id=wid,
        )
        self.assertIs(ws1, ws2)

        await self.mgr.close(wid)

    async def test_sandbox_metadata_injection(self) -> None:
        """Checkout should inject user/agent/workspace metadata."""
        ws = await self.mgr.create_workspace(
            user_id="test_user",
            agent_id="test_agent",
            session_id="s1",
        )
        wid = list(self.mgr._active.keys())[0]

        self.assertEqual(
            ws.sandbox_metadata.get("agentscope.user.id"),
            "test_user",
        )
        self.assertEqual(
            ws.sandbox_metadata.get("agentscope.agent.id"),
            "test_agent",
        )
        self.assertEqual(
            ws.sandbox_metadata.get("agentscope.workspace.id"),
            wid,
        )

        await self.mgr.close(wid)

    async def test_gateway_health(self) -> None:
        """Checked-out workspace has a healthy gateway."""
        ws = await self.mgr.create_workspace(
            user_id="u1",
            agent_id="a1",
            session_id="s1",
        )
        healthy = await ws.gateway_health()
        self.assertTrue(healthy)

        wid = list(self.mgr._active.keys())[0]
        await self.mgr.close(wid)

    async def test_close_all(self) -> None:
        """close_all drains every active workspace."""
        await self.mgr.create_workspace("u1", "a1", "s1")
        await self.mgr.create_workspace("u2", "a2", "s2")
        self.assertEqual(len(self.mgr._active), 2)

        await self.mgr.close_all()
        self.assertEqual(len(self.mgr._active), 0)


@unittest.skipUnless(_E2B_API_KEY, _SKIP_REASON)
class TestE2BPoolParameters(IsolatedAsyncioTestCase):
    """Verify pool construction parameters are wired correctly."""

    async def test_pool_params(self) -> None:
        """Pool internal state reflects constructor arguments."""
        mgr = E2BWorkspaceManager(
            api_key=_E2B_API_KEY,
            pool_enabled=True,
            pool_min_ready=2,
            pool_max_ready=4,
            pool_capacity=8,
            pool_batch_size=3,
        )
        pool = mgr._pool
        self.assertIsNotNone(pool)
        self.assertEqual(pool._pool_min_ready, 2)
        self.assertEqual(pool._pool_max_ready, 4)
        self.assertEqual(pool._pool_capacity, 8)
        self.assertEqual(pool._pool_batch_size, 3)
        self.assertEqual(pool._max_reuse, 1)
        self.assertIsNone(pool._reset_fn)

    async def test_pool_disabled(self) -> None:
        """pool_enabled=False should not create a pool."""
        mgr = E2BWorkspaceManager(
            api_key=_E2B_API_KEY,
            pool_enabled=False,
        )
        self.assertIsNone(mgr._pool)
        self.assertFalse(mgr._pool_enabled)


@unittest.skipUnless(_E2B_API_KEY, _SKIP_REASON)
class TestE2BPoolCallbacks(IsolatedAsyncioTestCase):
    """Verify pool callback functions work with real sandboxes."""

    async def test_factory_produces_live_workspace(self) -> None:
        """_pool_factory creates a running workspace."""
        mgr = E2BWorkspaceManager(
            api_key=_E2B_API_KEY,
            pool_enabled=True,
            pool_min_ready=0,
            pool_max_ready=1,
            pool_capacity=2,
            pool_batch_size=1,
        )
        ws = await mgr._pool_factory()
        try:
            self.assertTrue(ws.is_alive)
            self.assertIsNotNone(ws.workspace_id)
            healthy = await ws.gateway_health()
            self.assertTrue(healthy)
        finally:
            await ws.close()

    async def test_pause_resume_cycle(self) -> None:
        """pause + resume restores a working workspace."""
        mgr = E2BWorkspaceManager(
            api_key=_E2B_API_KEY,
            pool_enabled=True,
            pool_min_ready=0,
            pool_max_ready=1,
            pool_capacity=2,
            pool_batch_size=1,
        )
        ws = await mgr._pool_factory()
        try:
            self.assertTrue(ws.is_alive)

            await E2BWorkspaceManager._pool_pause(ws)
            self.assertFalse(ws.is_alive)

            await E2BWorkspaceManager._pool_resume(ws)
            self.assertTrue(ws.is_alive)
            healthy = await ws.gateway_health()
            self.assertTrue(healthy)
        finally:
            await ws.close()

    async def test_health_check(self) -> None:
        """_pool_health_check returns True for a live workspace."""
        mgr = E2BWorkspaceManager(
            api_key=_E2B_API_KEY,
            pool_enabled=True,
            pool_min_ready=0,
            pool_max_ready=1,
            pool_capacity=2,
            pool_batch_size=1,
        )
        ws = await mgr._pool_factory()
        try:
            result = await E2BWorkspaceManager._pool_health_check(ws)
            self.assertTrue(result)
        finally:
            await ws.close()

    async def test_close_destroys_workspace(self) -> None:
        """_pool_close destroys the workspace cleanly."""
        mgr = E2BWorkspaceManager(
            api_key=_E2B_API_KEY,
            pool_enabled=True,
            pool_min_ready=0,
            pool_max_ready=1,
            pool_capacity=2,
            pool_batch_size=1,
        )
        ws = await mgr._pool_factory()
        await E2BWorkspaceManager._pool_close(ws)
        self.assertFalse(ws.is_alive)


if __name__ == "__main__":
    unittest.main()
