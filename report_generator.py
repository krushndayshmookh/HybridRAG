"""
Report Generator for Hybrid RAG Evaluation
Creates comprehensive PDF/HTML reports with visualizations
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime
from matplotlib.backends.backend_pdf import PdfPages
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


class ReportGenerator:
    """
    Generate comprehensive evaluation reports with visualizations
    """
    
    def __init__(self, results_file="evaluation_results.json"):
        """Load evaluation results"""
        print("Loading evaluation results...")
        with open(results_file, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        
        self.results = self.data["detailed_results"]
        self.aggregated = self.data["aggregated_metrics"]
        self.innovative = self.data.get("innovative_evaluation", {})
        
        print(f"✓ Loaded results for {len(self.results)} questions")
    
    def create_metric_comparison_plot(self):
        """Create bar plot comparing main metrics"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        metrics = {
            'MRR\n(Retrieval)': self.aggregated['mean_mrr'],
            'ROUGE-L\n(Answer Quality)': self.aggregated['mean_rouge_l'],
            'NDCG@5\n(Ranking)': self.aggregated['mean_ndcg_at_5'],
            'Precision@5': self.aggregated['mean_precision_at_5'],
            'F1 Score': self.aggregated['mean_f1_score']
        }
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
        bars = ax.bar(metrics.keys(), metrics.values(), color=colors, alpha=0.8, edgecolor='black')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontweight='bold', fontsize=12)
        
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Evaluation Metrics Comparison', fontsize=14, fontweight='bold', pad=20)
        ax.set_ylim(0, 1.0)
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.3, label='Threshold (0.5)')
        ax.legend()
        
        plt.tight_layout()
        return fig
    
    def create_question_type_analysis(self):
        """Create grouped bar plot for metrics by question type"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        by_type = self.aggregated['by_question_type']
        
        question_types = list(by_type.keys())
        mrr_scores = [by_type[qt]['mean_mrr'] for qt in question_types]
        rouge_scores = [by_type[qt]['mean_rouge_l'] for qt in question_types]
        ndcg_scores = [by_type[qt]['mean_ndcg_at_5'] for qt in question_types]
        
        x = np.arange(len(question_types))
        width = 0.25
        
        ax.bar(x - width, mrr_scores, width, label='MRR', color='#FF6B6B', alpha=0.8)
        ax.bar(x, rouge_scores, width, label='ROUGE-L', color='#4ECDC4', alpha=0.8)
        ax.bar(x + width, ndcg_scores, width, label='NDCG@5', color='#45B7D1', alpha=0.8)
        
        ax.set_xlabel('Question Type', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Performance by Question Type', fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels([qt.capitalize() for qt in question_types])
        ax.legend()
        ax.set_ylim(0, 1.0)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def create_score_distributions(self):
        """Create distribution plots for main metrics"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        metrics_data = [
            ([r['mrr'] for r in self.results], 'MRR', '#FF6B6B'),
            ([r['rouge_l'] for r in self.results], 'ROUGE-L', '#4ECDC4'),
            ([r['ndcg_at_5'] for r in self.results], 'NDCG@5', '#45B7D1')
        ]
        
        for ax, (data, name, color) in zip(axes, metrics_data):
            ax.hist(data, bins=20, color=color, alpha=0.7, edgecolor='black')
            ax.axvline(np.mean(data), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(data):.3f}')
            ax.set_xlabel('Score', fontweight='bold')
            ax.set_ylabel('Frequency', fontweight='bold')
            ax.set_title(f'{name} Distribution', fontweight='bold')
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def create_ablation_study_plot(self):
        """Create ablation study comparison if available"""
        if 'ablation_study' not in self.innovative:
            return None
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ablation_data = self.innovative['ablation_study']['summary']
        
        methods = list(ablation_data.keys())
        mrr_scores = [ablation_data[m]['mean_mrr'] for m in methods]
        ndcg_scores = [ablation_data[m]['mean_ndcg'] for m in methods]
        
        x = np.arange(len(methods))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, mrr_scores, width, label='MRR', color='#FF6B6B', alpha=0.8)
        bars2 = ax.bar(x + width/2, ndcg_scores, width, label='NDCG@5', color='#4ECDC4', alpha=0.8)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=10)
        
        ax.set_xlabel('Retrieval Method', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Ablation Study: Retrieval Method Comparison', fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace('_', ' ').title() for m in methods])
        ax.legend()
        ax.set_ylim(0, max(max(mrr_scores), max(ndcg_scores)) * 1.2)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def create_error_analysis_plot(self):
        """Create error analysis visualization"""
        if 'error_analysis' not in self.innovative:
            return None
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        error_data = self.innovative['error_analysis']['failure_categories']
        
        categories = list(error_data.keys())
        counts = list(error_data.values())
        colors = ['#FF6B6B', '#FFA07A', '#FFD700', '#90EE90']
        
        bars = ax.bar(categories, counts, color=colors, alpha=0.8, edgecolor='black')
        
        # Add value labels and percentages
        total = sum(counts)
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            pct = (count / total * 100) if total > 0 else 0
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{count}\n({pct:.1f}%)',
                   ha='center', va='bottom', fontweight='bold')
        
        ax.set_xlabel('Error Category', fontsize=12, fontweight='bold')
        ax.set_ylabel('Count', fontsize=12, fontweight='bold')
        ax.set_title('Error Analysis: Failure Categories', fontsize=14, fontweight='bold', pad=20)
        ax.set_xticklabels([c.replace('_', ' ').title() for c in categories], rotation=15, ha='right')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def create_response_time_plot(self):
        """Create response time distribution"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        times = [r.get('response_time', 0) for r in self.results]
        
        ax.hist(times, bins=30, color='#96CEB4', alpha=0.7, edgecolor='black')
        ax.axvline(np.mean(times), color='red', linestyle='--', linewidth=2, 
                  label=f'Mean: {np.mean(times):.2f}s')
        ax.axvline(np.median(times), color='blue', linestyle='--', linewidth=2,
                  label=f'Median: {np.median(times):.2f}s')
        
        ax.set_xlabel('Response Time (seconds)', fontweight='bold')
        ax.set_ylabel('Frequency', fontweight='bold')
        ax.set_title('Response Time Distribution', fontweight='bold', fontsize=14, pad=20)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def create_correlation_heatmap(self):
        """Create correlation heatmap between metrics"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Extract metric columns
        df = pd.DataFrame(self.results)
        metric_cols = ['mrr', 'rouge_l', 'ndcg_at_5', 'precision_at_5', 'f1_score', 'exact_match']
        
        # Calculate correlation
        corr = df[metric_cols].corr()
        
        # Create heatmap
        sns.heatmap(corr, annot=True, fmt='.3f', cmap='coolwarm', 
                   square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                   vmin=-1, vmax=1, center=0, ax=ax)
        
        ax.set_title('Metric Correlation Matrix', fontsize=14, fontweight='bold', pad=20)
        
        # Better labels
        labels = ['MRR', 'ROUGE-L', 'NDCG@5', 'Prec@5', 'F1', 'EM']
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_yticklabels(labels, rotation=0)
        
        plt.tight_layout()
        return fig
    
    def generate_pdf_report(self, output_file="evaluation_report.pdf"):
        """Generate comprehensive PDF report"""
        print("\nGenerating PDF report...")
        
        with PdfPages(output_file) as pdf:
            # Page 1: Metric Comparison
            print("  Creating metric comparison...")
            fig = self.create_metric_comparison_plot()
            pdf.savefig(fig)
            plt.close(fig)
            
            # Page 2: Question Type Analysis
            print("  Creating question type analysis...")
            fig = self.create_question_type_analysis()
            pdf.savefig(fig)
            plt.close(fig)
            
            # Page 3: Score Distributions
            print("  Creating score distributions...")
            fig = self.create_score_distributions()
            pdf.savefig(fig)
            plt.close(fig)
            
            # Page 4: Ablation Study (if available)
            if 'ablation_study' in self.innovative:
                print("  Creating ablation study plot...")
                fig = self.create_ablation_study_plot()
                if fig:
                    pdf.savefig(fig)
                    plt.close(fig)
            
            # Page 5: Error Analysis (if available)
            if 'error_analysis' in self.innovative:
                print("  Creating error analysis plot...")
                fig = self.create_error_analysis_plot()
                if fig:
                    pdf.savefig(fig)
                    plt.close(fig)
            
            # Page 6: Response Time
            print("  Creating response time plot...")
            fig = self.create_response_time_plot()
            pdf.savefig(fig)
            plt.close(fig)
            
            # Page 7: Correlation Heatmap
            print("  Creating correlation heatmap...")
            fig = self.create_correlation_heatmap()
            pdf.savefig(fig)
            plt.close(fig)
            
            # Add metadata
            d = pdf.infodict()
            d['Title'] = 'Hybrid RAG System Evaluation Report'
            d['Author'] = 'Evaluation Pipeline'
            d['Subject'] = 'Performance Analysis'
            d['CreationDate'] = datetime.now()
        
        print(f"\n✓ PDF report saved to {output_file}")
    
    def generate_html_report(self, output_file="evaluation_report.html"):
        """Generate HTML report with summary statistics"""
        print("\nGenerating HTML report...")
        
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Hybrid RAG Evaluation Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
        }}
        .header h1 {{
            margin: 0;
            font-size: 2.5em;
        }}
        .header p {{
            margin: 10px 0 0 0;
            opacity: 0.9;
        }}
        .metric-card {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }}
        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .metric-value {{
            font-size: 3em;
            font-weight: bold;
            color: #667eea;
            margin: 10px 0;
        }}
        .metric-label {{
            font-size: 1.2em;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        .section {{
            background: white;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 30px;
        }}
        .section h2 {{
            color: #333;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
            margin-top: 0;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }}
        th {{
            background-color: #667eea;
            color: white;
            padding: 12px;
            text-align: left;
        }}
        td {{
            padding: 10px;
            border-bottom: 1px solid #ddd;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .good {{ color: #2ecc71; font-weight: bold; }}
        .medium {{ color: #f39c12; font-weight: bold; }}
        .poor {{ color: #e74c3c; font-weight: bold; }}
        .footer {{
            text-align: center;
            margin-top: 40px;
            padding: 20px;
            color: #666;
            border-top: 2px solid #ddd;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🚀 Hybrid RAG System Evaluation Report</h1>
        <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p>Total Questions Evaluated: {self.aggregated['total_questions']}</p>
    </div>
    
    <div class="metric-grid">
        <div class="metric-card">
            <div class="metric-label">Mean Reciprocal Rank</div>
            <div class="metric-value">{self.aggregated['mean_mrr']:.3f}</div>
            <div>± {self.aggregated['std_mrr']:.3f}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">ROUGE-L Score</div>
            <div class="metric-value">{self.aggregated['mean_rouge_l']:.3f}</div>
            <div>± {self.aggregated['std_rouge_l']:.3f}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">NDCG@5 Score</div>
            <div class="metric-value">{self.aggregated['mean_ndcg_at_5']:.3f}</div>
            <div>± {self.aggregated['std_ndcg_at_5']:.3f}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Exact Match Rate</div>
            <div class="metric-value">{self.aggregated['exact_match_rate']:.3f}</div>
            <div>Success Rate</div>
        </div>
    </div>
    
    <div class="section">
        <h2>📊 Performance by Question Type</h2>
        <table>
            <thead>
                <tr>
                    <th>Question Type</th>
                    <th>Count</th>
                    <th>MRR</th>
                    <th>ROUGE-L</th>
                    <th>NDCG@5</th>
                </tr>
            </thead>
            <tbody>
"""
        
        for qtype, stats in self.aggregated['by_question_type'].items():
            html_content += f"""
                <tr>
                    <td><strong>{qtype.capitalize()}</strong></td>
                    <td>{stats['count']}</td>
                    <td class="{'good' if stats['mean_mrr'] > 0.7 else 'medium' if stats['mean_mrr'] > 0.4 else 'poor'}">{stats['mean_mrr']:.3f}</td>
                    <td class="{'good' if stats['mean_rouge_l'] > 0.6 else 'medium' if stats['mean_rouge_l'] > 0.3 else 'poor'}">{stats['mean_rouge_l']:.3f}</td>
                    <td class="{'good' if stats['mean_ndcg_at_5'] > 0.7 else 'medium' if stats['mean_ndcg_at_5'] > 0.4 else 'poor'}">{stats['mean_ndcg_at_5']:.3f}</td>
                </tr>
"""
        
        html_content += """
            </tbody>
        </table>
    </div>
"""
        
        # Add ablation study if available
        if 'ablation_study' in self.innovative:
            ablation = self.innovative['ablation_study']['summary']
            html_content += """
    <div class="section">
        <h2>🔬 Ablation Study Results</h2>
        <table>
            <thead>
                <tr>
                    <th>Method</th>
                    <th>Mean MRR</th>
                    <th>Mean NDCG@5</th>
                    <th>Questions Tested</th>
                </tr>
            </thead>
            <tbody>
"""
            for method, stats in ablation.items():
                html_content += f"""
                <tr>
                    <td><strong>{method.replace('_', ' ').title()}</strong></td>
                    <td>{stats['mean_mrr']:.3f}</td>
                    <td>{stats['mean_ndcg']:.3f}</td>
                    <td>{stats['count']}</td>
                </tr>
"""
            html_content += """
            </tbody>
        </table>
    </div>
"""
        
        # Add error analysis if available
        if 'error_analysis' in self.innovative:
            error_data = self.innovative['error_analysis']['failure_categories']
            total = sum(error_data.values())
            html_content += """
    <div class="section">
        <h2>⚠️ Error Analysis</h2>
        <table>
            <thead>
                <tr>
                    <th>Error Category</th>
                    <th>Count</th>
                    <th>Percentage</th>
                </tr>
            </thead>
            <tbody>
"""
            for category, count in error_data.items():
                pct = (count / total * 100) if total > 0 else 0
                html_content += f"""
                <tr>
                    <td><strong>{category.replace('_', ' ').title()}</strong></td>
                    <td>{count}</td>
                    <td>{pct:.1f}%</td>
                </tr>
"""
            html_content += """
            </tbody>
        </table>
    </div>
"""
        
        html_content += """
    <div class="footer">
        <p><strong>Hybrid RAG System Evaluation</strong></p>
        <p>This report was automatically generated by the evaluation pipeline.</p>
        <p>For detailed visualizations, see evaluation_report.pdf</p>
    </div>
</body>
</html>
"""
        
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(html_content)
        
        print(f"✓ HTML report saved to {output_file}")
    
    def generate_all_reports(self):
        """Generate both PDF and HTML reports"""
        print("\n" + "="*70)
        print("GENERATING COMPREHENSIVE REPORTS")
        print("="*70)
        
        self.generate_pdf_report("evaluation_report.pdf")
        self.generate_html_report("evaluation_report.html")
        
        print("\n" + "="*70)
        print("REPORT GENERATION COMPLETE!")
        print("="*70)
        print("\nGenerated files:")
        print("  📄 evaluation_report.pdf  - Visualizations and charts")
        print("  🌐 evaluation_report.html - Interactive summary")
        print("="*70 + "\n")


def main():
    """Main function to generate reports"""
    import os
    
    if not os.path.exists("evaluation_results.json"):
        print("Error: evaluation_results.json not found!")
        print("Please run evaluation_pipeline.py first.")
        return
    
    generator = ReportGenerator("evaluation_results.json")
    generator.generate_all_reports()


if __name__ == "__main__":
    main()
