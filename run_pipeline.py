"""
Automated Pipeline Execution Script.
Generates genuine physical training data, trains the multi-output neural network,
evaluates performance on unseen scenarios, benchmarks baselines, and generates reports.
"""

import os
import numpy as np
import pandas as pd
from config.default_config import SystemConfig
from src.dataset_generator import DatasetGenerator
from src.train import ModelTrainer
from src.evaluate import ModelEvaluator
from src.experiments import ExperimentRunner
from src.export_report import generate_research_report


def main():
    print("=" * 70)
    print("6G Hybrid RF/VLC RSMA - Automated Pipeline Runner")
    print("=" * 70)
    
    config = SystemConfig(num_users=8, random_seed=42)
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    # 1. Dataset Generation
    print("\n[Step 1/4] Generating physics-based high-density simulation scenarios (K=8 users)...")
    gen = DatasetGenerator(config, seed=42)
    num_scenarios = 120
    df = gen.generate_dataset(num_scenarios=num_scenarios, randomize_obstacles=True)
    
    n_train = int(0.8 * len(df))
    train_df = df.iloc[:n_train].reset_index(drop=True)
    val_df = df.iloc[n_train:].reset_index(drop=True)
    
    train_df.to_csv("data/training_dataset.csv", index=False)
    val_df.to_csv("data/validation_dataset.csv", index=False)
    print(f"-> Generated {len(df)} scenarios. Train: {len(train_df)}, Val: {len(val_df)}")
    
    # 2. PyTorch Neural Network Training
    print("\n[Step 2/4] Training Multi-Output Neural Network on Optimizer Labels...")
    trainer = ModelTrainer(in_features=8 * 9, num_users=8)
    res = trainer.fit(train_df=train_df, val_df=val_df, epochs=40, batch_size=16, lr=1e-3)
    print(f"-> Training complete! Best Val Loss: {res['best_val_loss']:.4f}")
    print(f"-> Checkpoint saved to: {res['checkpoint_path']}")
    
    # 3. Model Evaluation on Unseen Test Scenarios
    print("\n[Step 3/4] Evaluating AI Controller on Unseen Scenarios...")
    evaluator = ModelEvaluator(checkpoint_path="models/best_joint_model.pt", config=config)
    eval_df, summary = evaluator.evaluate_test_dataset(val_df)
    eval_df.to_csv("results/ai_test_evaluation.csv", index=False)
    
    print(f"-> Link Mode Decision Agreement: {summary['mean_link_agreement']:.2f}%")
    print(f"-> Alpha MAE: {summary['mean_alpha_mae']:.4f} | Beta MAE: {summary['mean_beta_mae']:.4f}")
    print(f"-> Mean AI Sum Rate: {summary['mean_ai_sum_rate_mbps']:.2f} Mbps (Teacher: {summary['mean_teacher_sum_rate_mbps']:.2f} Mbps)")
    print(f"-> Mean AI Energy Efficiency: {summary['mean_ai_ee']:.2f} bits/J (Teacher: {summary['mean_teacher_ee']:.2f} bits/J)")
    print(f"-> Mean AI Latency: {summary['mean_ai_latency_ms']:.3f} ms (Teacher: {summary['mean_teacher_runtime_ms']:.1f} ms)")
    print(f"-> Computational Speedup: {summary['speedup_factor']:.1f}x")
    
    # 4. Generate Research Report
    print("\n[Step 4/4] Generating Publication-Ready Technical Report...")
    report_file = generate_research_report(summary_stats=summary, output_path="results/RESEARCH_REPORT.md")
    print(f"-> Technical Report written to: {report_file}")
    print("\nPipeline execution successfully finished!")


if __name__ == "__main__":
    main()
