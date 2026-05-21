# -*- coding: utf-8 -*-
"""Configuration models for workspace backends."""

from pydantic import BaseModel, Field


class MCPServerConfig(BaseModel):
    """One MCP server to manage inside a workspace.

    Supports two transport protocol types:

    - ``"stdio"`` (default): spawn a local process via command + args.
    - ``"http"``: connect to an HTTP MCP server (SSE or Streamable HTTP).

    For stdio: ``command`` (and optionally ``args``) are required.
    For http: ``url`` is required; ``headers`` and ``timeout`` are optional.
    """

    name: str
    protocol: str = "stdio"  # "stdio" | "http"

    # stdio fields
    command: str = ""
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)

    # http fields
    url: str = ""
    headers: dict[str, str] = Field(default_factory=dict)
    timeout: float = 30.0
