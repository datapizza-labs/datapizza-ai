from datapizza.tools.mcptoolbox import MCPToolBoxTool
from datapizza.tools import Tool

# Configuration
URL = "http://localhost:5000"

# Clients for the tests
try:
    print(f"\n ---- Create 3 agents initializing MCPToolBoxTool with URL: {URL} ----")
    client_1 = MCPToolBoxTool(toolbox_url=URL)
    print("\n ---- Agent 1: created ----")
    client_2 = MCPToolBoxTool(toolbox_url=URL, tool_names=["search_artists"])
    print("\n ---- Agent 2: created ----")
    client_3 = MCPToolBoxTool(
        toolbox_url=URL, tool_names=["search_artists"], toolset_names=["neo4j-tools"]
    )
    print("\n ---- Agent 3: created ----")
except Exception as e:
    print(f"ERROR during testing: {e}")


try:
    # --- Test 1: load_single_tool ---
    print("\n--- Test 1: load_single_tool('search_artists') ---")
    tool1: Tool = client_2.load_single_tool("search_artists")

    print(f"Type: {type(tool1)}")
    print(f"Name: {tool1.name}")
    print(f"Description: {tool1.description}")
    # print(f"Parameters: {tool1.parameters}") # Deprecated/Old name? Use schema or similar if needed

    # --- Test 3: load_tools() (Automatic discovery) ---
    print("\n--- Test 3: load_tools() (Automatic discovery) ---")
    # Create a new factory without specifying names
    all_discovered = client_1.load_tools()
    print(f"Discovered {len(all_discovered)} tools automatically for client 1.")
    all_discovered = client_2.load_tools()
    print(f"Discovered {len(all_discovered)} tools automatically for client 2.")
    all_discovered = client_3.load_tools()
    print(f"Discovered {len(all_discovered)} tools automatically for client 3.")

    # --- Test 4: info() ---
    print("\n--- Test 4: info() for client 1 ---")
    tools_info = client_1.list_tools()
    for name, info in tools_info.items():
        print(f"\nTool: {name}")
        print(f" - Description: {info['description']}")
        print(f" - Required Params: {info['required']}")
    print("\n--- Test 4: info() for client 2 ---")
    tools_info = client_2.list_tools()
    for name, info in tools_info.items():
        print(f"\nTool: {name}")
        print(f" - Description: {info['description']}")
        print(f" - Required Params: {info['required']}")
    print("\n--- Test 4: info() for client 3 ---")
    tools_info = client_3.list_tools()
    for name, info in tools_info.items():
        print(f"\nTool: {name}")
        print(f" - Description: {info['description']}")
        print(f" - Required Params: {info['required']}")


except Exception as e:
    print(f"\nERROR during testing: {e}")


finally:
    # --- Cleanup ---
    print("\n--- Cleaning up resources ---")
    client_1.close()
    client_2.close()
    client_3.close()

print("\n--- Testing complete ---")
