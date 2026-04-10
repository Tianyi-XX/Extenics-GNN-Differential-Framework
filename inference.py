import json
import torch
from pathlib import Path
from train import HeterogeneousMultiTaskGNN
from torch_geometric.data import Data

# ============ Multi-Domain Analysis ============
class MultiDomainConflictAnalyzer:
    def __init__(self, model_path='best_model_structural.pt', device='cpu'):
        self.device = torch.device(device)

        self.domain_mapping = {
            'ecological': 0,
            'financial': 1,
            'medical': 2,
            'production': 3
        }
        self.domain_names = {
            0: 'ecological',
            1: 'financial',
            2: 'medical',
            3: 'production'
        }

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

        self.relation_mapping = checkpoint['relation_type_mapping']
        edge_dim = checkpoint['edge_dim']
        self.num_relation_types = len(self.relation_mapping)

        print(f"✓ Loaded relation mapping with {self.num_relation_types} types from checkpoint.")
        print(f"✓ Model edge dimension set to {edge_dim} from checkpoint.")

        self.model = HeterogeneousMultiTaskGNN(
            in_channels=6,
            hidden_channels=128,
            edge_dim=edge_dim,
            num_domains=4
        ).to(self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        print("✓ Multi-domain model loaded successfully")
        print(f"  Best validation loss: {checkpoint['val_loss']:.4f}")

    def analyze_problem(self, data_dict: dict, domain: str,
                        top_k_nodes: int = 10,
                        conflict_threshold: float = 0.3) -> dict:
        """
        Analyzes a new problem and returns key information
        """
        if domain not in self.domain_mapping:
            raise ValueError(f"Unknown domain: {domain}. Must be one of {list(self.domain_mapping.keys())}")

        graph = self._build_graph(data_dict, domain)
        graph = graph.to(self.device)

        with torch.no_grad():
            outputs = self.model(graph)

        analysis = self._parse_predictions(
            data_dict, outputs, graph, top_k_nodes, conflict_threshold
        )

        return analysis

    def _build_graph(self, data_dict, domain):
        """Converts a data dictionary into a PyG graph object"""

        domain_id = self.domain_mapping[domain]

        node_ids = []
        node_types = []
        node_id_to_idx = {}
        node_id_to_info = {}

        for m in data_dict.get('matter_elements', []):
            idx = len(node_ids)
            node_ids.append(m['id'])
            node_types.append(0)
            node_id_to_idx[m['id']] = idx
            node_id_to_info[m['id']] = {
                'type': 'matter',
                'name': m.get('name', f"matter_{idx}"),
                'features': m.get('features', {})
            }

        for a in data_dict.get('action_elements', []):
            idx = len(node_ids)
            node_ids.append(a['id'])
            node_types.append(1)
            node_id_to_idx[a['id']] = idx
            node_id_to_info[a['id']] = {
                'type': 'action',
                'action': a.get('action', f"action_{idx}"),
                'features': a.get('features', {})
            }

        num_nodes = len(node_ids)
        x = torch.zeros((num_nodes, 6))
        for i, node_type in enumerate(node_types):
            x[i, node_type] = 1
            x[i, 2 + domain_id] = 1

        edge_index = []
        edge_attr = []
        edge_info = []

        for rel in data_dict.get('relation_elements', []):
            src_idx = node_id_to_idx.get(rel['source'])
            tgt_idx = node_id_to_idx.get(rel['target'])

            if src_idx is not None and tgt_idx is not None:
                edge_index.append([src_idx, tgt_idx])

                edge_type_vec = [0] * (self.num_relation_types + 4)
                rel_type = rel.get('relation', '').strip()

                ## Normalize relation type variants for consistency with training vocabulary
                if rel_type == "Conflict":
                    rel_type = "Conflicts"

                # Use .get() method, fall back to '<UNK>' if not found
                relation_index = self.relation_mapping.get(rel_type, self.relation_mapping['<UNK>'])
                edge_type_vec[relation_index] = 1

                edge_type_vec[self.num_relation_types + domain_id] = 1
                edge_attr.append(edge_type_vec)

                edge_info.append({
                    'id': rel['id'],
                    'type': rel_type,
                    'source': rel['source'],
                    'target': rel['target'],
                    'features': rel.get('features', {})
                })

        graph = Data(
            x=x,
            edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous() if edge_index else torch.empty(
                (2, 0), dtype=torch.long),
            edge_attr=torch.tensor(edge_attr, dtype=torch.float),
            num_nodes=num_nodes,
        )
        graph.domain_id = torch.tensor([domain_id], dtype=torch.long)
        graph.domain_name = domain
        graph.node_ids = node_ids
        graph.node_info = node_id_to_info
        graph.edge_info = edge_info

        # Map single graph to batch format required by the model
        if num_nodes > 0:
            graph.batch = torch.zeros(num_nodes, dtype=torch.long)

        return graph

    def _parse_predictions(self, data_dict, outputs, graph, top_k, threshold):
        """Parses prediction results and extracts key information"""

        # 1. Extract Top-K important nodes
        importance_scores = outputs['node_importance'].cpu().numpy()
        relevance_scores = outputs['node_relevance'].cpu().numpy()

        combined_scores = importance_scores * relevance_scores

        # Ensure top_k does not exceed the actual number of nodes
        num_actual_nodes = len(graph.node_ids)
        top_k = min(top_k, num_actual_nodes)

        top_k_indices = combined_scores.argsort()[-top_k:][::-1]

        key_nodes = []
        for idx in top_k_indices:
            node_id = graph.node_ids[idx]
            node_info = graph.node_info[node_id]

            key_nodes.append({
                'id': node_id,
                'name': node_info.get('name') or node_info.get('action'),
                'type': node_info['type'],
                'importance': float(importance_scores[idx]),
                'relevance': float(relevance_scores[idx]),
                'combined_score': float(combined_scores[idx]),
                'features': node_info['features']
            })

        # 2. Extract conflict edges
        conflicts = []
        if graph.edge_attr.shape[0] > 0:  # Process only if edges exist
            conflict_probs = outputs['edge_is_conflict'].cpu().numpy()
            severity_scores = outputs['edge_severity'].cpu().numpy()
            problem_probs = outputs['edge_is_problem'].cpu().numpy()

            for i, edge_info in enumerate(graph.edge_info):
                is_conflict = conflict_probs[i] > threshold

                if is_conflict:
                    src_info = graph.node_info[edge_info['source']]
                    tgt_info = graph.node_info[edge_info['target']]

                    conflicts.append({
                        'id': edge_info['id'],
                        'source': {
                            'id': edge_info['source'],
                            'name': src_info.get('name') or src_info.get('action')
                        },
                        'target': {
                            'id': edge_info['target'],
                            'name': tgt_info.get('name') or tgt_info.get('action')
                        },
                        'relation_type': edge_info['type'],
                        'conflict_probability': float(conflict_probs[i]),
                        'severity': float(severity_scores[i]),
                        'is_problem': float(problem_probs[i]),
                        'description': edge_info['features'].get('description', '')
                    })

            conflicts.sort(key=lambda x: x['severity'], reverse=True)

        # 3. Collect importance distribution for all nodes
        all_nodes_importance = []
        for idx, node_id in enumerate(graph.node_ids):
            node_info = graph.node_info[node_id]
            all_nodes_importance.append({
                'id': node_id,
                'name': node_info.get('name') or node_info.get('action'),
                'importance': float(importance_scores[idx]),
                'relevance': float(relevance_scores[idx])
            })

        return {
            'domain': graph.domain_name,
            'key_nodes': key_nodes,
            'conflicts': conflicts,
            'all_nodes_importance': all_nodes_importance,
            'summary': self._generate_summary(key_nodes, conflicts, graph.domain_name),
            'num_nodes': graph.num_nodes,
            'num_edges': graph.num_edges
        }

    def _generate_summary(self, key_nodes, conflicts, domain):
        """Generates an analysis summary"""
        domain_labels = {
            'ecological': 'Ecological Domain',
            'financial': 'Financial Domain',
            'medical': 'Medical Domain',
            'production': 'Production Domain'
        }

        return {
            'domain_label': domain_labels.get(domain, domain),
            'total_key_nodes': len(key_nodes),
            'total_conflicts': len(conflicts),
            'avg_conflict_severity': sum(c['severity'] for c in conflicts) / len(conflicts) if conflicts else 0,
            'max_conflict_severity': max([c['severity'] for c in conflicts]) if conflicts else 0,
            'most_critical_conflict': conflicts[0] if conflicts else None,
            'high_severity_conflicts': len([c for c in conflicts if c['severity'] > 0.7]),
            'avg_key_node_importance': sum(n['importance'] for n in key_nodes) / len(key_nodes) if key_nodes else 0
        }


# ============ Enhanced LLM Prompt Generator ============
class EnhancedLLMPromptGenerator:
    """
    Converts GNN structural analysis results into structured natural language prompts for LLM solution generation,
    following the template defined in Appendix A of the paper.
    """
    # Domain context configurations for prompt generation.
    # To add a new domain, append an entry following the same structure:
    # 'domain_key': {
    #     'label': 'Display Name',
    #     'focus': 'Key analytical focus areas for this domain',
    #     'methods': 'Recommended innovation methodologies'
    # }
    DOMAIN_CONTEXTS = {
        'ecological': {
            'label': 'Ecological Environment',
            'focus': 'Ecological balance, environmental protection, sustainable development',
            'methods': 'TRIZ Theory, Ecosystem Analysis, Circular Economy Principles'
        },
        'financial': {
            'label': 'Financial Economy',
            'focus': 'Risk management, resource allocation, market equilibrium',
            'methods': 'TRIZ Theory, Game Theory, System Dynamics'
        },
        'medical': {
            'label': 'Medical & Health',
            'focus': 'Diagnosis/treatment optimization, patient safety, medical resource allocation',
            'methods': 'TRIZ Theory, Clinical Decision Support, Evidence-Based Medicine'
        },
        'production': {
            'label': 'Production & Manufacturing',
            'focus': 'Process optimization, quality control, efficiency improvement',
            'methods': 'TRIZ Theory, Lean Production, Six Sigma'
        }
    }

    @staticmethod
    def generate_prompt(analysis: dict, original_data: dict,
                        include_all_nodes: bool = False,
                        prompt_style: str = 'detailed') -> str:
        domain = analysis['domain']
        context = EnhancedLLMPromptGenerator.DOMAIN_CONTEXTS.get(domain, {})
        summary = analysis['summary']

        # Section A.1: Domain Contextualization Header
        prompt = f"""# [{context.get('label', domain)}] Intelligent Technical Problem Analysis and Solution Generation

## Problem Overview
**Domain**: {context.get('label', domain)}
**Analysis Focus**: {context.get('focus', 'Problem identification and resolution')}
**Recommended Methods**: {context.get('methods', 'TRIZ Theory, System Analysis')}

## Basic Problem Information
{EnhancedLLMPromptGenerator._format_problem_description(original_data)}
---
## Graph Neural Network Analysis Results

## Analysis Summary
- **Key Elements Identified**: {summary['total_key_nodes']} core nodes
- **Conflicts Discovered**: {summary['total_conflicts']}
- **Average Conflict Severity**: {summary['avg_conflict_severity']:.2%}
- **Maximum Conflict Severity**: {summary['max_conflict_severity']:.2%}
- **High-Severity Conflicts**: {summary['high_severity_conflicts']} (Severity > 0.7)
---
## Core Element Identification (Top-{len(analysis['key_nodes'])})
"""
        # Section A.2: Key Element Enumeration
        for i, node in enumerate(analysis['key_nodes'], 1):
            prompt += f"{i}. **{node['name']}**\n"
            prompt += f"   - Type: {'Matter-element' if node['type'] == 'matter' else 'Action-element'}\n"
            prompt += f"   - Importance Score: {node['importance']:.3f}\n"
            prompt += f"   - Problem Relevance: {node['relevance']:.3f}\n"
            prompt += f"   - Combined Score: {node['combined_score']:.3f}\n"
            if node['features'] and prompt_style == 'detailed':
                prompt += f"   - Features: {EnhancedLLMPromptGenerator._format_features(node['features'])}\n"
            prompt += "\n"

        # Section A.2: Conflict Relation Specification
        prompt += "\n---\n\n## Conflict Analysis\n\n"
        if len(analysis['conflicts']) == 0:
            prompt += "No significant conflict relations detected.\n\n"
        else:
            prompt += f"A total of **{len(analysis['conflicts'])}** conflicts were identified, sorted by severity:\n\n"
            for i, conflict in enumerate(analysis['conflicts'][:15], 1):
                severity_level = EnhancedLLMPromptGenerator._get_severity_level(conflict['severity'])
                prompt += f"### Conflict #{i} ({severity_level['label']})\n"
                prompt += f"- **Severity**: {conflict['severity']:.3f}\n"
                prompt += f"- **Conflict Probability**: {conflict['conflict_probability']:.2%}\n"
                prompt += f"- **Is Problem**: {conflict['is_problem']:.2%}\n"
                prompt += f"- **Source Node**: {conflict['source']['name']}\n"
                prompt += f"- **Target Node**: {conflict['target']['name']}\n"
                prompt += f"- **Relation Type**: {conflict['relation_type']}\n"
                if conflict['description']:
                    prompt += f"- **Detailed Description**: {conflict['description']}\n"
                prompt += "\n"

        if include_all_nodes and prompt_style == 'detailed':
            prompt += "\n---\n\n### Complete Node Importance Distribution\n\n"
            sorted_nodes = sorted(analysis['all_nodes_importance'],
                                  key=lambda x: x['importance'], reverse=True)
            prompt += "| Node Name | Importance | Relevance |\n|-----------|------------|-----------|\n"
            for node in sorted_nodes[:20]:
                prompt += f"| {node['name']} | {node['importance']:.3f} | {node['relevance']:.3f} |\n"
            prompt += "\n"

        # Section A.3: Generation Task Directives
        prompt += """
---
## Solution Generation Task
Based on the AI GNN analysis above, complete the following:
### 1. Root Cause Analysis
- Synthesize key elements and conflicts to identify root causes
- Distinguish primary from secondary contradictions
- Trace causal chains and dependencies
### 2. Solution Design
"""
        if len(analysis['conflicts']) > 0:
            prompt += f"- Propose targeted strategies to resolve the {len(analysis['conflicts'])} identified conflicts\n"
            prompt += f"- Prioritize high-severity conflicts (severity > 0.7, Total: {summary['high_severity_conflicts']})\n"

        prompt += f"""- Leverage Top-{len(analysis['key_nodes'])} key element characteristics
- Respect domain-specific constraints of the {context.get('label', domain)} domain
### 3. Innovative Breakthrough
- Apply Extenics transformation methods
- Propose breakthrough ideas beyond conventional thinking
- Consider technological evolution paths
### 4. Implementation Planning
- Provide phased implementation steps
- Assess expected outcomes and risks
- Identify critical success factors
---
## Output Requirements
Provide structured output including:
1. **Problem Diagnosis** - Root cause and contradiction analysis
2. **Solutions** - At least 3 specific, feasible solutions
3. **Novelty Suggestions** - Breakthrough ideas
4. **Implementation Path** - Step-by-step plan
5. **Impact Assessment** - Expected outcomes and risks

Ensure solutions are:
(v) **Targeted** - Address GNN-identified conflicts and key elements
(v) **Feasible** - Consider practical constraints
(v) **Novel** - Apply Extenics transformations
(v) **Systematic** - Account for element interactions
"""
        return prompt

    @staticmethod
    def _format_problem_description(data: dict) -> str:
        desc = f"""**Problem Scale**:
- Number of Matter-elements: {len(data.get('matter_elements', []))}
- Number of Action-elements: {len(data.get('action_elements', []))}
- Number of Relation-elements: {len(data.get('relation_elements', []))}
"""
        if 'description' in data:
            desc += f"\n**Problem Description**: {data['description']}\n"
        return desc

    @staticmethod
    def _format_features(features: dict) -> str:
        if not features:
            return "None"
        items = [f"{k}: {v}" for k, v in features.items()]
        return '; '.join(items)

    @staticmethod
    def _get_severity_level(severity: float) -> dict:
        if severity >= 0.9:
            return {'label': 'Critical'}
        elif severity >= 0.7:
            return {'label': 'Severe'}
        elif severity >= 0.5:
            return {'label': 'Medium'}
        else:
            return {'label': 'Minor'}


# ============ Batch Analysis Tool ============
class BatchAnalyzer:
    def __init__(self, analyzer: MultiDomainConflictAnalyzer):
        self.analyzer = analyzer

    def analyze_directory(self, data_dir: Path, domain: str,
                          output_dir: Path, top_k: int = 10):
        output_dir.mkdir(exist_ok=True)
        data_files = sorted(data_dir.glob('*_data.json'))
        print(f"\nBatch analyzing {len(data_files)} files from {domain} domain...")
        results = []
        for data_file in data_files:
            print(f"  Processing {data_file.name}...", end=' ')
            try:
                with open(data_file, 'r', encoding='utf-8') as f:
                    problem_data = json.load(f)
                analysis = self.analyzer.analyze_problem(problem_data, domain, top_k_nodes=top_k)
                prompt = EnhancedLLMPromptGenerator.generate_prompt(analysis, problem_data, prompt_style='detailed')
                example_name = data_file.stem.replace('_data', '')
                analysis_file = output_dir / f"{example_name}_analysis.json"
                with open(analysis_file, 'w', encoding='utf-8') as f:
                    json.dump(analysis, f, ensure_ascii=False, indent=2)
                prompt_file = output_dir / f"{example_name}_prompt.txt"
                with open(prompt_file, 'w', encoding='utf-8') as f:
                    f.write(prompt)
                results.append({'file': data_file.name, 'status': 'success', 'conflicts': len(analysis['conflicts']),
                                'key_nodes': len(analysis['key_nodes'])})
                print("✓")
            except Exception as e:
                print(f"✗ Error: {e}")
                results.append({'file': data_file.name, 'status': 'failed', 'error': str(e)})

        report = {'domain': domain, 'total_files': len(data_files),
                  'successful': len([r for r in results if r['status'] == 'success']),
                  'failed': len([r for r in results if r['status'] == 'failed']), 'results': results}
        report_file = output_dir / f"batch_analysis_report_{domain}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(
            f"\n Batch analysis completed!\n  Successful: {report['successful']}/{report['total_files']}\n  Report saved to: {report_file}")
        return report


# ============ Main Entry Point ============
def main():
    import argparse
    parser = argparse.ArgumentParser(description='Multi-domain GNN Conflict Analysis')
    parser.add_argument('--mode', type=str, default='single', choices=['single', 'batch'],
                        help='Analysis mode: single file or batch')
    parser.add_argument('--domain', type=str, help='Problem domain. Required for single/batch mode.')
    parser.add_argument('--data_file', type=str, help='Path to data file (for single mode)')
    parser.add_argument('--data_dir', type=str, help='Path to data directory (for batch mode)')
    parser.add_argument('--output_dir', type=str, default='./analysis_results', help='Output directory for results')
    parser.add_argument('--model_path', type=str, default='best_model_structural.pt', help='Path to trained model')
    parser.add_argument('--top_k', type=int, default=10, help='Number of top key nodes to extract')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'],
                        help='Device to use for inference')
    args = parser.parse_args()

    # If in command-line mode, domain is required
    if args.domain is None:
        print("Error: the --domain argument is required for command-line usage.")
        parser.print_help()
        return

    analyzer = MultiDomainConflictAnalyzer(model_path=args.model_path, device=args.device)

    if args.mode == 'single':
        if not args.data_file:
            print("Error: --data_file is required for single mode")
            return
        print(f"\nAnalyzing single file: {args.data_file}\nDomain: {args.domain}\n" + "=" * 70)
        with open(args.data_file, 'r', encoding='utf-8') as f:
            problem_data = json.load(f)
        analysis = analyzer.analyze_problem(problem_data, args.domain, top_k_nodes=args.top_k)

        print("\n[KEY NODES]")
        for node in analysis['key_nodes']:
            print(f"  - {node['name']}: Importance={node['importance']:.3f}, Relevance={node['relevance']:.3f}")
        print(f"\n[CONFLICTS IDENTIFIED] (Total: {len(analysis['conflicts'])})")
        for conflict in analysis['conflicts'][:10]:
            print(f"  - {conflict['source']['name']} → {conflict['target']['name']}: Severity={conflict['severity']:.3f}")

        print("\n" + "=" * 70 + "\nGenerating LLM Prompt...\n" + "=" * 70)
        prompt = EnhancedLLMPromptGenerator.generate_prompt(analysis, problem_data, prompt_style='detailed')

        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)
        analysis_file = output_dir / 'analysis.json'
        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, ensure_ascii=False, indent=2)
        prompt_file = output_dir / 'llm_prompt.txt'
        with open(prompt_file, 'w', encoding='utf-8') as f:
            f.write(prompt)

        print(f"\nResults saved to {output_dir}\n  - Analysis: {analysis_file}\n  - Prompt: {prompt_file}")
        print("\n[Prompt Preview]\n" + prompt[:800] + "\n...(Omitted)")

    elif args.mode == 'batch':
        if not args.data_dir:
            print("Error: --data_dir is required for batch mode")
            return
        batch_analyzer = BatchAnalyzer(analyzer)
        report = batch_analyzer.analyze_directory(Path(args.data_dir), args.domain, Path(args.output_dir) / args.domain,
                                                  top_k=args.top_k)

    print("\n" + "=" * 70 + "\nAnalysis completed!\n" + "=" * 70)



if __name__ == "__main__":
    main()