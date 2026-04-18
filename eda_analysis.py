import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.power import TTestIndPower

class StudentPerformanceAnalyzer:
    def __init__(self, filepath, output_dir='output'):
        self.filepath = filepath
        self.output_dir = output_dir
        self.df = None
        os.makedirs(self.output_dir, exist_ok=True)
        sns.set_theme(style="whitegrid")

    def load_data(self):
        self.df = pd.read_csv(self.filepath)
        
        with open(f'{self.output_dir}/data_profile.txt', 'w') as f:
            f.write(f"Shape: {self.df.shape}\n\n")
            f.write(str(self.df.dtypes) + "\n\n")
            f.write(str(self.df.isnull().sum()) + "\n\n")
            f.write(str(self.df.describe()))
        return self

    def clean_data(self):
        if 'commute_minutes' in self.df.columns:
            self.df['commute_minutes'] = self.df['commute_minutes'].fillna(self.df['commute_minutes'].median())
        
        if 'scholarship' in self.df.columns:
            self.df['scholarship'] = self.df['scholarship'].fillna('None')
        return self

    def plot_visuals(self):
        plt.figure(figsize=(10, 6))
        sns.histplot(self.df['gpa'], kde=True, color='skyblue')
        plt.savefig(f'{self.output_dir}/gpa_distribution.png')
        
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=self.df, x='department', y='gpa', hue='department', palette='Set3', legend=False)
        plt.xticks(rotation=45)
        plt.savefig(f'{self.output_dir}/gpa_by_dept.png')

        plt.figure(figsize=(10, 8))
        numeric_df = self.df.select_dtypes(include=[np.number])
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
        plt.savefig(f'{self.output_dir}/correlation_heatmap.png')
        
        plt.close('all')
        return self

    def run_statistics(self):
        groups = [group['gpa'].values for name, group in self.df.groupby('department')]
        f_stat, p_val_anova = stats.f_oneway(*groups)
        
        analysis = TTestIndPower()
        required_n = analysis.solve_power(effect_size=0.5, power=0.8, alpha=0.05)
        
        bootstrap_means = [self.df['gpa'].sample(frac=1, replace=True).mean() for _ in range(10000)]
        ci_lower, ci_upper = np.percentile(bootstrap_means, [2.5, 97.5])
        
        print(f"ANOVA P-Value: {p_val_anova:.4f}")
        print(f"Required N: {required_n:.2f}")
        print(f"Bootstrap CI: ({ci_lower:.4f}, {ci_upper:.4f})")
        return self

    def auto_report(self):
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            plt.figure(figsize=(8, 4))
            sns.kdeplot(self.df[col], fill=True)
            plt.savefig(f'{self.output_dir}/auto_{col}.png')
            plt.close()

if __name__ == "__main__":
    PATH = 'data/student_performance.csv'
    
    if os.path.exists(PATH):
        analyzer = StudentPerformanceAnalyzer(PATH)
        (analyzer.load_data()
                 .clean_data()
                 .plot_visuals()
                 .run_statistics()
                 .auto_report())