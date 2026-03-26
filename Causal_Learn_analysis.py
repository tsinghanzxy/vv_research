# -*- coding: utf-8 -*-
"""
Causal Structure Discovery Pipeline for Vibrio vulnificus Pathogenicity
========================================================================
FIXED VERSION: Corrected pydot attribute methods
Alpha = 0.05
"""

import os
import sys
import pandas as pd
import numpy as np
import glob
import argparse
import warnings
from typing import Optional, List

# Causal Discovery Libraries
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.GraphUtils import GraphUtils
from causallearn.utils.cit import chisq

# Visualization
import pydot

warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")
warnings.filterwarnings("ignore", category=FutureWarning, module="causallearn")

# --- Configuration Constants ---
DEFAULT_ALPHA = 0.05
DEFAULT_TOP_N = 20
TARGET_MODEL = 'lightgbm'

def parse_arguments():
    parser = argparse.ArgumentParser(description="Causal Discovery Pipeline for V. vulnificus")
    parser.add_argument('--results_dir', type=str, default='1212VV_kfold_results_27_models_*')
    parser.add_argument('--pan_genome', type=str, default='1212VV_gene_presence_absence_min')
    parser.add_argument('--metadata', type=str, default='1212VV_metadata_min')
    parser.add_argument('--pca_file', type=str, default=None)
    parser.add_argument('--alpha', type=float, default=DEFAULT_ALPHA)
    parser.add_argument('--top_n', type=int, default=DEFAULT_TOP_N)
    parser.add_argument('--output_prefix', type=str, default='causal_graph_vv')
    return parser.parse_args()

def find_latest_results_dir(pattern: str) -> Optional[str]:
    dirs = glob.glob(pattern)
    if not dirs:
        return None
    return max(dirs, key=os.path.getmtime)

def load_data_adaptive(base_name: str, is_pangenome: bool = False) -> pd.DataFrame:
    feather_path = f"{base_name}.feather"
    xlsx_path = f"{base_name}.xlsx"
    
    df = None
    if os.path.exists(feather_path):
        print(f"  [IO] Loading feather: {feather_path}")
        df = pd.read_feather(feather_path)
    elif os.path.exists(xlsx_path):
        print(f"  [IO] Loading excel: {xlsx_path}")
        df = pd.read_excel(xlsx_path)
    else:
        raise FileNotFoundError(f"Data file not found: {base_name} (.feather or .xlsx)")
    
    potential_index_cols = ['index', 'Strain_Name', 'Strain', 'BioSample', '菌株名', 'ID', 'Isolate']
    set_index = False
    
    if isinstance(df.index, pd.RangeIndex):
        for col in potential_index_cols:
            if col in df.columns:
                print(f"  [Index] Setting '{col}' as index")
                df = df.set_index(col)
                set_index = True
                break
        if not set_index and 'index' in df.columns:
            print(f"  [Index] Setting 'index' column as index")
            df = df.set_index('index')
            set_index = True
            
    if not set_index:
        print(f"  [Warning] No obvious index column found. Using current index.")
    
    df.index = df.index.astype(str).str.strip()
    df.columns = df.columns.astype(str).str.strip()
    
    return df

def prepare_causal_dataset(results_dir: str, model_name: str, top_n: int, 
                           pan_base: str, meta_base: str, 
                           pca_file: Optional[str] = None) -> Optional[pd.DataFrame]:
    print(f"\n{'='*60}")
    print(f"STEP 1: Data Preparation & Feature Selection")
    print(f"{'='*60}")
    
    possible_paths = [
        os.path.join(results_dir, model_name.lower().replace(" ", "_"), "aggregated", "aggregated_shap_feature_importance.csv"),
        os.path.join(results_dir, model_name.lower(), "aggregated", "aggregated_shap_feature_importance.csv")
    ]
    shap_path = next((p for p in possible_paths if os.path.exists(p)), None)
    
    if not shap_path:
        print(f"  [Error] SHAP file not found.")
        return None
        
    shap_df = pd.read_csv(shap_path)
    if 'Feature' not in shap_df.columns:
        print(f"  [Error] 'Feature' column not found in SHAP file.")
        return None
        
    top_features = shap_df.head(top_n)['Feature'].astype(str).str.strip().tolist()
    print(f"  [Select] Top {len(top_features)} features based on {model_name} SHAP values")
    
    try:
        pan_df = load_data_adaptive(pan_base, is_pangenome=True)
        meta_df = load_data_adaptive(meta_base, is_pangenome=False)
    except Exception as e:
        print(f"  [Error] Data loading failed: {e}")
        return None
    
    common_indices = meta_df.index.intersection(pan_df.index)
    print(f"  [Match] Common strains found: {len(common_indices)}")
    
    if len(common_indices) == 0:
        print("  [CRITICAL ERROR] No matching strain names between metadata and pangenome!")
        return None
        
    meta_df = meta_df.loc[common_indices]
    pan_df = pan_df.loc[common_indices]
    
    existing_features = [f for f in top_features if f in pan_df.columns]
    print(f"  [Match] {len(existing_features)}/{len(top_features)} features found in pangenome")
    
    if len(existing_features) == 0:
        print("  [Error] No SHAP features found in pangenome matrix.")
        return None
    
    merged_df = pd.concat([meta_df, pan_df[existing_features]], axis=1)
    print(f"  [Merge] Merged shape: {merged_df.shape}")
    
    phenotype_col = None
    possible_names = ['菌株表型', 'Phenotype', 'Label', 'Source', 'Isolation_Source', '类型']
    for col in possible_names:
        if col in merged_df.columns:
            phenotype_col = col
            break
    
    if not phenotype_col:
        for col in merged_df.columns:
            if 'phenotype' in col.lower() or 'source' in col.lower() or '表型' in col:
                phenotype_col = col
                break
                
    if not phenotype_col:
        print(f"  [Error] Phenotype column not found.")
        return None
        
    print(f"  [Info] Using phenotype column: '{phenotype_col}'")
    
    label_map = {}
    raw_labels = merged_df[phenotype_col].astype(str).str.strip()
    for val in raw_labels.unique():
        if val.lower() in ['clinical', 'clinical isolate', 'human', 'patient', '1']:
            label_map[val] = 1
        elif val.lower() in ['environmental', 'environment', 'water', 'oyster', '0']:
            label_map[val] = 0
        else:
            label_map[val] = np.nan
            
    merged_df['Label'] = raw_labels.map(label_map)
    
    if merged_df['Label'].isna().all():
        print("  [Error] Failed to map phenotype values to 0/1.")
        return None
        
    if pca_file and os.path.exists(pca_file):
        try:
            pca_df = pd.read_csv(pca_file, index_col=0)
            pca_df.index = pca_df.index.astype(str).str.strip()
            common_all = merged_df.index.intersection(pca_df.index)
            merged_df = merged_df.loc[common_all]
            pca_df = pca_df.loc[common_all]
            
            pc_cols = [c for c in ['PC1', 'PC2', 'PC3'] if c in pca_df.columns]
            if pc_cols:
                merged_df = pd.concat([merged_df, pca_df[pc_cols]], axis=1)
                existing_features.extend(pc_cols)
                print(f"  [Confounder] Added {pc_cols}")
        except Exception as e:
            print(f"  [Warning] PCA loading failed: {e}")
    
    analysis_cols = existing_features + ['Label']
    for col in analysis_cols:
        merged_df[col] = pd.to_numeric(merged_df[col], errors='coerce')
        
    raw_count = len(merged_df)
    clean_df = merged_df[analysis_cols].dropna()
    print(f"  [Clean] Samples: {raw_count} -> {len(clean_df)}")
    
    if len(clean_df) < 10:
        print(f"  [Warning] Too few samples ({len(clean_df)})")
        return None
        
    return clean_df

def run_pc_algorithm(data_df: pd.DataFrame, alpha: float) -> Optional[object]:
    print(f"\n{'='*60}")
    print(f"STEP 2: Causal Structure Discovery (PC Algorithm)")
    print(f"{'='*60}")
    print(f"  [Config] Alpha = {alpha}, Test = Chi-Square (Discrete Data)")
    
    data_np = data_df.to_numpy().astype(float)
    node_names = [str(col) for col in data_df.columns]
    
    if np.isnan(data_np).any():
        print("  [Error] NaNs detected in numpy array.")
        return None
        
    try:
        cg = pc(data_np, alpha=alpha, independence_test=chisq, show_progress=True)
        print("  [Success] PC algorithm completed.")
        return cg, node_names
    except Exception as e:
        print(f"  [Error] PC algorithm failed: {e}")
        return None

def visualize_and_export(cg, node_names: List[str], output_prefix: str, data_df: pd.DataFrame):
    """
    FIXED: Corrected pydot attribute methods
    """
    print(f"\n{'='*60}")
    print(f"STEP 3: Visualization & Export (Optimized)")
    print(f"{'='*60}")
    
    try:
        pydot_graph = GraphUtils.to_pydot(cg.G, labels=node_names)
    except Exception as e:
        print(f"  [Error] Graph conversion failed: {e}")
        return

    # Identify Target Node (Label)
    target_id = None
    possible_names = ['Label', 'label', '菌株表型', 'Phenotype']
    node_map = {str(n.get_name()): str(n.get_attributes().get('label', n.get_name())).strip('"') for n in pydot_graph.get_nodes()}
    
    for nid, name in node_map.items():
        if name in possible_names:
            target_id = nid
            break
    
    if not target_id and node_map:
        target_id = list(node_map.keys())[0]
        print(f"  [Warning] Target label not found. Using first node: {node_map[target_id]}")
    
    # Extract Edges for CSV Export
    edges_data = []
    direct_influencers = []
    
    for edge in pydot_graph.get_edges():
        u = str(edge.get_source())
        v = str(edge.get_destination())
        u_name = node_map.get(u, u)
        v_name = node_map.get(v, v)
        
        edges_data.append({
            'Source_ID': u, 'Source_Name': u_name,
            'Target_ID': v, 'Target_Name': v_name,
            'To_Phenotype': (v == target_id)
        })
        
        if v == target_id:
            direct_influencers.append(u)
    
    edges_df = pd.DataFrame(edges_data)
    edges_df.to_csv(f"{output_prefix}_edges.csv", index=False)
    print(f"  [Save] Edge list saved to {output_prefix}_edges.csv")
    print(f"  [Info] Direct influencers: {[node_map.get(u, u) for u in direct_influencers]}")
    
    # ========== LAYOUT OPTIMIZATION ==========
    pydot_graph.set_graph_defaults(
        rankdir='LR',
        splines='ortho',
        concentrate='false',
        nodesep='0.8',
        ranksep='1.5',
        fontname='Arial',
        fontsize='10',
        bgcolor='white'
    )

    # ========== NODE STYLE OPTIMIZATION (FIXED) ==========
    for node in pydot_graph.get_nodes():
        nid = node.get_name()
        # Use set() method for pydot compatibility
        node.set_style('filled, rounded')
        
        if nid == target_id:
            # Phenotype node: Gold, large
            node.set_fillcolor('#FFD700')
            node.set_color('#B8860B')
            node.set_fontsize('16')
            node.set_fontcolor('#000000')
            node.set_penwidth('4.0')
            node.set_width('2.0')
            node.set_height('1.0')
            # FIXED: Use fontweight as attribute, not method
            node.set('fontweight', 'bold')
            print(f"  [Style] Phenotype node styled: {node_map.get(nid, nid)}")
        elif nid in direct_influencers:
            # Direct causal influencers: Light blue
            node.set_fillcolor('#AEC7E8')
            node.set_color('#1F77B4')
            node.set_fontsize('12')
            node.set_fontcolor('#000000')
            node.set_penwidth('2.5')
            node.set('fontweight', 'bold')
        else:
            # Other genes: Gray, semi-transparent
            node.set_fillcolor('#F7F7F7')
            node.set_color('#D3D3D3')
            node.set_fontsize('9')
            node.set_fontcolor('#999999')
            node.set_penwidth('1.0')
            node.set('opacity', '0.7')

    # ========== EDGE STYLE OPTIMIZATION ==========
    red_count = 0
    blue_count = 0
    gray_count = 0
    
    for edge in pydot_graph.get_edges():
        v_id = str(edge.get_destination())
        u_id = str(edge.get_source())
        
        edge.set_arrowhead('vee')
        edge.set_arrowsize('1.0')
        
        if v_id == target_id:
            edge.set_color('#D62728')
            edge.set_penwidth('4.0')
            edge.set_style('bold')
            edge.set_arrowsize('1.8')
            red_count += 1
        elif u_id in direct_influencers or v_id in direct_influencers:
            edge.set_color('#1F77B4')
            edge.set_penwidth('2.5')
            edge.set_style('solid')
            blue_count += 1
        else:
            edge.set_color('#D3D3D3')
            edge.set_penwidth('0.8')
            edge.set_style('dashed')
            edge.set('opacity', '0.5')
            gray_count += 1

    print(f"  [Style] Edge statistics: Red={red_count}, Blue={blue_count}, Gray={gray_count}")

    # ========== EXPORT FILES ==========
    try:
        with open(f'{output_prefix}.dot', 'w', encoding='utf-8') as f:
            f.write(pydot_graph.to_string())
        print(f"  [Save] DOT file: {output_prefix}.dot")
        
        pydot_graph.write_png(f'{output_prefix}.png')
        print(f"  [Save] PNG image: {output_prefix}.png")
        
        pydot_graph.write_pdf(f'{output_prefix}_2.pdf')
        print(f"  [Save] PDF vector: {output_prefix}_2.pdf")
        
        pydot_graph.write_svg(f'{output_prefix}.svg')
        print(f"  [Save] SVG vector: {output_prefix}.svg")
        
        print(f"\n{'='*60}")
        print(f"[COMPLETE] Causal analysis finished successfully!")
        print(f"{'='*60}")
        print(f"  Output prefix: {output_prefix}")
        print(f"  Direct causal drivers: {[node_map.get(u, u) for u in direct_influencers]}")
        print(f"  Recommendation: Use {output_prefix}_2.pdf for manuscript")
        
    except Exception as e:
        print(f"  [Error] Export failed: {e}")
        print(f"  [Hint] Ensure Graphviz is installed and 'dot' is in PATH")

def main():
    args = parse_arguments()
    
    results_dir = find_latest_results_dir(args.results_dir)
    if not results_dir:
        print(f"[Error] No results directory found matching: {args.results_dir}")
        sys.exit(1)
    
    print(f"[Info] Using results directory: {results_dir}")
    folder_tag = os.path.basename(results_dir)
    
    causal_df = prepare_causal_dataset(
        results_dir=results_dir,
        model_name=TARGET_MODEL,
        top_n=args.top_n,
        pan_base=args.pan_genome,
        meta_base=args.metadata,
        pca_file=args.pca_file
    )
    
    if causal_df is None or causal_df.empty:
        print("[Error] Data preparation failed. Exiting.")
        sys.exit(1)
        
    result = run_pc_algorithm(causal_df, alpha=args.alpha)
    
    if result:
        cg, node_names = result
        output_prefix = f"{args.output_prefix}_{folder_tag}_alpha{str(args.alpha).replace('.', '')}"
        visualize_and_export(cg, node_names, output_prefix, causal_df)

if __name__ == "__main__":
    main()