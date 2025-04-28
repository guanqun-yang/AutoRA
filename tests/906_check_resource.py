import re
import pandas as pd
import subprocess

def parse_tres(tres_string):
    """Parse TRES like 'cpu=64,mem=257440M,billing=64,gres/gpu=2' into a dict."""
    tres = {}
    for item in tres_string.split(','):
        if '=' in item:
            key, value = item.split('=')
            tres[key.strip()] = value.strip()
    return tres

def normalize_mem(val):
    """Normalize memory values into MB."""
    val = val.upper()
    if val.endswith('G'):
        return int(val[:-1]) * 1024
    elif val.endswith('M'):
        return int(val[:-1])
    else:
        return int(val)

def get_scontrol_show_node() -> str:
    """Run `scontrol show node` and capture its output."""
    result = subprocess.run(["scontrol", "show", "node"], stdout=subprocess.PIPE, text=True)
    return result.stdout

def parse_scontrol_to_df(scontrol_text: str) -> pd.DataFrame:
    """Parse scontrol show node raw text output into a pandas DataFrame."""
    nodes = scontrol_text.split("NodeName=")[1:]

    parsed_nodes = []
    for node_text in nodes:
        node_text = "NodeName=" + node_text.replace("\n", " ")
        pairs = re.findall(r"(\S+?)=(\S+)", node_text)
        node_info = dict(pairs)
        parsed_nodes.append(node_info)

    df = pd.DataFrame(parsed_nodes)
    return df.astype(str)

def is_node_fully_available(cfg_tres_str: str, alloc_tres_str: str) -> bool:
    """Check if node is fully available based on CfgTRES and AllocTRES."""
    if not cfg_tres_str:
        return False

    cfg_tres = parse_tres(cfg_tres_str)
    alloc_tres = parse_tres(alloc_tres_str) if alloc_tres_str else {}

    for key in cfg_tres:
        cfg_value = cfg_tres[key]
        alloc_value = alloc_tres.get(key)

        if 'mem' in key:
            cfg_value = normalize_mem(cfg_value)
            alloc_value = normalize_mem(alloc_value) if alloc_value else 0
        else:
            cfg_value = int(cfg_value)
            alloc_value = int(alloc_value) if alloc_value else 0

        if alloc_value >= cfg_value:
            return False

    return True

def check_available_nodes_from_df(df: pd.DataFrame):
    """Check availability of nodes from a parsed DataFrame."""
    available_nodes = []

    for idx, row in df.iterrows():
        node_name = row.get('NodeName')
        cfg_tres = row.get('CfgTRES')
        alloc_tres = row.get('AllocTRES', '').strip()

        if is_node_fully_available(cfg_tres, alloc_tres):
            available_nodes.append(node_name)

    if available_nodes:
        print("\n✅ Fully available node(s):")
        for node in available_nodes:
            print(f"  - {node}")
    else:
        print("\n❌ No fully available nodes (ALL resources available).")

def main():
    scontrol_text = get_scontrol_show_node()
    df = parse_scontrol_to_df(scontrol_text)
    check_available_nodes_from_df(df)

if __name__ == "__main__":
    main()