# -*- coding: utf-8 -*-
# pylint: disable=protected-access
"""Integration tests for :class:`DockerWorkspaceManager` pool mode.

Every test in this module creates real Docker containers via the
workspace manager's pool. The whole module is skipped when no Docker
daemon is reachable.
"""

import io
import os
import shutil
import subprocess
import tarfile
import tempfile
import unittest
import uuid
from unittest.async_case import IsolatedAsyncioTestCase

from agentscope.app.workspace_manager._docker_workspace_manager import (
    DockerWorkspaceManager,
)
from agentscope.app.workspace_manager._workspace_pool import (
    PooledState,
)

# ── docker daemon detection ────────────────────────────────────────


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
_SKIP_REASON = "Docker daemon not available"


# ── tests ──────────────────────────────────────────────────────────


@unittest.skipUnless(_DOCKER_OK, _SKIP_REASON)
class TestDockerPoolLifecycle(IsolatedAsyncioTestCase):
    """Pool lifecycle: start, acquire, release, stop with real containers."""

    async def asyncSetUp(self) -> None:
        self.mgr = DockerWorkspaceManager(
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

        wid = ws.workspace_id
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


@unittest.skipUnless(_DOCKER_OK, _SKIP_REASON)
class TestDockerPoolParameters(IsolatedAsyncioTestCase):
    """Verify pool construction parameters are wired correctly."""

    async def test_pool_params(self) -> None:
        """Pool internal state reflects constructor arguments."""
        mgr = DockerWorkspaceManager(
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
        mgr = DockerWorkspaceManager(pool_enabled=False)
        self.assertIsNone(mgr._pool)
        self.assertFalse(mgr._pool_enabled)


@unittest.skipUnless(_DOCKER_OK, _SKIP_REASON)
class TestDockerPoolHostSync(IsolatedAsyncioTestCase):
    """Verify host-workdir ↔ container sync in pool mode."""

    async def asyncSetUp(self) -> None:
        # pylint: disable=consider-using-with
        self.tmpdir = tempfile.TemporaryDirectory()
        self.mgr = DockerWorkspaceManager(
            basedir=self.tmpdir.name,
            pool_enabled=True,
            pool_min_ready=1,
            pool_max_ready=1,
            pool_capacity=3,
            pool_batch_size=1,
        )
        await self.mgr.__aenter__()

    async def asyncTearDown(self) -> None:
        await self.mgr.__aexit__(None, None, None)
        self.tmpdir.cleanup()

    async def test_sync_host_to_container_on_checkout(self) -> None:
        """Files in host workdir are uploaded into the container on get."""
        user_id, agent_id = "u1", "a1"
        host_workdir = os.path.join(self.tmpdir.name, user_id, agent_id)
        os.makedirs(host_workdir, exist_ok=True)
        with open(os.path.join(host_workdir, "seed.txt"), "w") as f:
            f.write("hello from host")

        ws = await self.mgr.create_workspace(
            user_id=user_id,
            agent_id=agent_id,
            session_id="s1",
        )

        # Verify the file is visible inside the container via the
        # workspace's exec mechanism.
        result = await ws._exec("cat /workspace/seed.txt", timeout=5)
        self.assertIn(b"hello from host", result.stdout)

        wid = list(self.mgr._active.keys())[0]
        await self.mgr.close(wid)

    async def test_sync_container_to_host_on_release(self) -> None:
        """Files created inside the container land on the host after close."""
        user_id, agent_id = "u1", "a1"

        ws = await self.mgr.create_workspace(
            user_id=user_id,
            agent_id=agent_id,
            session_id="s1",
        )

        # Create a file inside the container.
        await ws._exec(
            "echo 'created inside' > /workspace/new_file.txt",
            timeout=5,
        )

        wid = list(self.mgr._active.keys())[0]
        await self.mgr.close(wid)

        # The file should now exist on the host.
        host_workdir = os.path.join(self.tmpdir.name, user_id, agent_id)
        host_file = os.path.join(host_workdir, "new_file.txt")
        self.assertTrue(os.path.isfile(host_file))
        with open(host_file) as f:
            content = f.read().strip()
        self.assertEqual(content, "created inside")


@unittest.skipUnless(_DOCKER_OK, _SKIP_REASON)
class TestDockerPoolCallbacks(IsolatedAsyncioTestCase):
    """Verify pool callback functions work with real workspaces."""

    async def test_factory_produces_live_workspace(self) -> None:
        """_pool_factory creates a running workspace with no workdir."""
        mgr = DockerWorkspaceManager(
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
            self.assertIsNone(ws.workdir)
            healthy = await ws.gateway_health()
            self.assertTrue(healthy)
        finally:
            await ws.close()

    async def test_pause_resume_cycle(self) -> None:
        """pause + resume restores a working workspace."""
        mgr = DockerWorkspaceManager(
            pool_enabled=True,
            pool_min_ready=0,
            pool_max_ready=1,
            pool_capacity=2,
            pool_batch_size=1,
        )
        ws = await mgr._pool_factory()
        try:
            self.assertTrue(ws.is_alive)

            await DockerWorkspaceManager._pool_pause(ws)
            self.assertFalse(ws.is_alive)

            await DockerWorkspaceManager._pool_resume(ws)
            self.assertTrue(ws.is_alive)
            healthy = await ws.gateway_health()
            self.assertTrue(healthy)
        finally:
            await ws.close()

    async def test_health_check(self) -> None:
        """_pool_health_check returns True for a live workspace."""
        mgr = DockerWorkspaceManager(
            pool_enabled=True,
            pool_min_ready=0,
            pool_max_ready=1,
            pool_capacity=2,
            pool_batch_size=1,
        )
        ws = await mgr._pool_factory()
        try:
            result = await DockerWorkspaceManager._pool_health_check(ws)
            self.assertTrue(result)
        finally:
            await ws.close()

    async def test_close_destroys_workspace(self) -> None:
        """_pool_close destroys the workspace cleanly."""
        mgr = DockerWorkspaceManager(
            pool_enabled=True,
            pool_min_ready=0,
            pool_max_ready=1,
            pool_capacity=2,
            pool_batch_size=1,
        )
        ws = await mgr._pool_factory()
        await DockerWorkspaceManager._pool_close(ws)
        self.assertFalse(ws.is_alive)


if __name__ == "__main__":
    unittest.main()
