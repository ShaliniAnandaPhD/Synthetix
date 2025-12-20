"""
mcp_agent.py - Model Context Protocol Client Agent

An agent that dynamically discovers and uses tools from MCP servers.
Enables plug-and-play integration with data sources and APIs.
"""

import logging
import re
from typing import Dict, Any, List, Optional

import requests

from .reflex_agent import ReflexAgent

logger = logging.getLogger(__name__)


class MCPAgent(ReflexAgent):
    """
    An agent that connects to MCP servers and dynamically discovers tools.
    
    Follows the Model Context Protocol pattern for standardized
    tool discovery and invocation.
    
    Usage:
        agent = MCPAgent(name="DataAgent", mcp_server_url="http://localhost:8000")
        
        # Discover tools from server
        agent.discover_tools()
        
        # Call a tool directly
        result = agent.call_tool("get_live_score", {"team_name": "Chiefs"})
        
        # Or via message processing
        result = agent.process("MCP_CALL:get_live_score|team_name=Chiefs")
        result = agent.process("SCORE:Chiefs")  # If score tool discovered
    """
    
    def __init__(
        self,
        name: str = "MCPAgent",
        mcp_server_url: str = "http://localhost:8000",
        auto_discover: bool = True,
        **kwargs
    ):
        """
        Initialize the MCP Agent.
        
        Args:
            name: Agent name
            mcp_server_url: Base URL of the MCP server
            auto_discover: Whether to discover tools on init
            **kwargs: Additional args passed to ReflexAgent
        """
        super().__init__(name=name, **kwargs)
        
        self.mcp_server_url = mcp_server_url.rstrip('/')
        self.discovered_tools: Dict[str, Dict[str, Any]] = {}
        self.discovered_resources: Dict[str, str] = {}
        
        # Add the MCP_CALL rule
        self.add_rule("mcp_call", self._mcp_call_rule)
        
        # Auto-discover tools from server
        if auto_discover:
            try:
                self._discover_tools()
                self._discover_resources()
            except Exception as e:
                logger.warning(f"Auto-discovery failed: {e}")
        
        logger.info(f"MCPAgent '{name}' initialized with server: {mcp_server_url}")
    
    def _discover_tools(self):
        """
        Discover available tools from the MCP server.
        
        Calls GET /mcp/tools and registers dynamic rules for each tool.
        """
        try:
            response = requests.get(f"{self.mcp_server_url}/mcp/tools", timeout=5)
            response.raise_for_status()
            data = response.json()
            
            tools = data.get("tools", [])
            logger.info(f"Discovered {len(tools)} tools from MCP server")
            
            for tool in tools:
                tool_name = tool.get("name")
                endpoint = tool.get("endpoint")
                description = tool.get("description", "")
                arguments = tool.get("arguments", {})
                
                self.discovered_tools[tool_name] = {
                    "endpoint": endpoint,
                    "description": description,
                    "arguments": arguments
                }
                
                # Register shortcut rules for common tools
                self._register_tool_shortcuts(tool_name)
                
                logger.info(f"  📦 Tool: {tool_name} -> {endpoint}")
            
        except requests.RequestException as e:
            logger.error(f"Failed to discover tools: {e}")
            raise
    
    def _discover_resources(self):
        """Discover available resources from the MCP server."""
        try:
            response = requests.get(f"{self.mcp_server_url}/mcp/resources", timeout=5)
            response.raise_for_status()
            data = response.json()
            
            resources = data.get("resources", [])
            logger.info(f"Discovered {len(resources)} resources")
            
            for resource in resources:
                uri = resource.get("uri")
                endpoint = resource.get("endpoint")
                self.discovered_resources[uri] = endpoint
                logger.info(f"  📊 Resource: {uri}")
                
        except requests.RequestException as e:
            logger.warning(f"Failed to discover resources: {e}")
    
    def _register_tool_shortcuts(self, tool_name: str):
        """Register shortcut rules for common tool patterns."""
        
        if tool_name == "get_live_score":
            # Register "SCORE:TeamName" pattern
            def score_rule(msg):
                content = msg.content if hasattr(msg, 'content') else str(msg)
                if content.upper().startswith("SCORE:"):
                    team = content[6:].strip()
                    return self.call_tool("get_live_score", {"team_name": team})
                return {"skipped": True}
            
            self.add_rule("score_shortcut", score_rule)
            logger.debug("Registered SCORE: shortcut rule")
        
        elif tool_name == "get_player_stats":
            # Register "STATS:PlayerName" pattern
            def stats_rule(msg):
                content = msg.content if hasattr(msg, 'content') else str(msg)
                if content.upper().startswith("STATS:"):
                    player = content[6:].strip()
                    return self.call_tool("get_player_stats", {"player_name": player})
                return {"skipped": True}
            
            self.add_rule("stats_shortcut", stats_rule)
            logger.debug("Registered STATS: shortcut rule")
    
    def _mcp_call_rule(self, msg) -> Dict[str, Any]:
        """
        Rule handler for MCP_CALL messages.
        
        Format: MCP_CALL:tool_name|arg1=value1|arg2=value2
        """
        content = msg.content if hasattr(msg, 'content') else str(msg)
        
        if not content.upper().startswith("MCP_CALL:"):
            return {"skipped": True}
        
        try:
            # Parse: MCP_CALL:tool_name|arg1=val1|arg2=val2
            payload = content[9:]  # Remove "MCP_CALL:"
            parts = payload.split("|")
            
            tool_name = parts[0].strip()
            args = {}
            
            for part in parts[1:]:
                if "=" in part:
                    key, value = part.split("=", 1)
                    args[key.strip()] = value.strip()
            
            return self.call_tool(tool_name, args)
            
        except Exception as e:
            logger.error(f"MCP_CALL parse error: {e}")
            return {"error": str(e), "status": "failed"}
    
    def call_tool(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call a tool on the MCP server.
        
        Args:
            tool_name: Name of the tool to call
            args: Arguments to pass to the tool
            
        Returns:
            Tool result dictionary
        """
        if tool_name not in self.discovered_tools:
            return {"error": f"Unknown tool: {tool_name}", "status": "failed"}
        
        tool_info = self.discovered_tools[tool_name]
        endpoint = tool_info["endpoint"]
        
        try:
            logger.info(f"🔧 Calling MCP tool: {tool_name}({args})")
            
            response = requests.post(
                f"{self.mcp_server_url}{endpoint}",
                params=args,
                timeout=10
            )
            response.raise_for_status()
            result = response.json()
            
            logger.info(f"✅ Tool result: {result}")
            return {
                "tool": tool_name,
                "result": result,
                "status": "success"
            }
            
        except requests.RequestException as e:
            logger.error(f"Tool call failed: {e}")
            return {"error": str(e), "tool": tool_name, "status": "failed"}
    
    def get_resource(self, uri: str) -> Dict[str, Any]:
        """
        Fetch a resource from the MCP server.
        
        Args:
            uri: Resource URI (e.g., "nfl://scores/all")
            
        Returns:
            Resource data dictionary
        """
        if uri not in self.discovered_resources:
            return {"error": f"Unknown resource: {uri}"}
        
        endpoint = self.discovered_resources[uri]
        
        try:
            response = requests.get(f"{self.mcp_server_url}{endpoint}", timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return {"error": str(e)}
    
    def list_tools(self) -> List[str]:
        """List all discovered tools."""
        return list(self.discovered_tools.keys())
    
    def list_resources(self) -> List[str]:
        """List all discovered resources."""
        return list(self.discovered_resources.keys())
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get agent information including MCP details."""
        base_info = super().get_agent_info() if hasattr(super(), 'get_agent_info') else {}
        base_info.update({
            "agent_type": "MCPAgent",
            "mcp_server": self.mcp_server_url,
            "tools_discovered": len(self.discovered_tools),
            "resources_discovered": len(self.discovered_resources),
            "capabilities": ["mcp_tools", "dynamic_discovery"]
        })
        return base_info
