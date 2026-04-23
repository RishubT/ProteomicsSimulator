#TODO:
'''

- Add Numpy 2.2 and python 3.10 is needed 
Change alpha meaning in documentation
change uniform distribution to normal distribution
- Not for STD to avoid negitive std

- Change STD to be for an array that the user inputs for each cell type
- Change  cell proportions between sample to make it more realistic

'''
import numpy as np
from typing import Union, List
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

@dataclass
class SimulationResults:
    bulk_sample: np.ndarray
    latent_expression_profiles: np.ndarray
    cell_proportions: np.ndarray
    protein_to_cell_map: np.ndarray
    cell_type_means: np.ndarray
    protein_cell_means: np.ndarray
    
    def display_box_plots(self, sample = None):
        """
        Boxplot 1: One box for all proteins, then one box per cell type (marker proteins only),
                with a global median dashed line.
        Boxplot 2: One box per marker protein across all samples, each with a unique color,
                labeled by protein index.
        """
        bulk = self.bulk_sample  # (n_proteins, n_samples)
        marker_map = self.protein_to_cell_map  # (n_proteins,)
        n_cell_types = self.cell_proportions.shape[1]

        # ── Boxplot 1: All proteins + marker proteins by cell type ───────
        fig1, ax1 = plt.subplots(figsize=(max(6, (n_cell_types + 1) * 2), 5))

        box_data = [bulk.flatten()]
        box_labels = ["All proteins"]

        for i in range(n_cell_types):
            marker_idx = np.where(marker_map == i)[0]
            box_data.append(bulk[marker_idx, :].flatten())
            box_labels.append(f"Type {i + 1}")

        color = "#e8a0a0"
        bp1 = ax1.boxplot(
            box_data,
            patch_artist=True,
            medianprops=dict(color="black", linewidth=1.5),
        )
        for patch in bp1["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.8)

        # Global median line
        global_mean = np.mean(bulk)
        ax1.axhline(
            y=global_mean,
            color="blue",
            linestyle="--",
            linewidth=1,
            label=f"Global Mean ({global_mean:.2f})",
        )

        global_median = np.median(bulk)
        ax1.axhline(
            y=global_median,
            color="red",
            linestyle="--",
            linewidth=1,
            label=f"Global Median ({global_median:.2f})",
        )

        ax1.set_xticklabels(box_labels)
        ax1.set_xlabel("Category")
        ax1.set_ylabel("Bulk intensity")
        ax1.set_title("Cell Abundance Distribution: all proteins vs. markers by cell type")
        ax1.legend(loc="upper right")
        fig1.tight_layout()

        # ── Boxplot 2: Each marker protein across samples ────────────────
        marker_indices = np.where(marker_map != -1)[0]
        n_markers = len(marker_indices)

        fig2, ax2 = plt.subplots(figsize=(max(8, n_markers * 0.8), 5))

        marker_data = [bulk[p, :] for p in marker_indices]
        marker_labels = [str(p) for p in marker_indices]

        # Color by cell type assignment
        cmap = plt.cm.get_cmap("tab10", n_cell_types)
        box_colors = [cmap(marker_map[p]) for p in marker_indices]

        bp2 = ax2.boxplot(
            marker_data,
            patch_artist=True,
            medianprops=dict(color="black", linewidth=1.5),
        )
        for patch, c in zip(bp2["boxes"], box_colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.8)

        ax2.set_xticks(range(1, n_markers + 1))
        ax2.set_xticklabels(marker_labels, rotation=45, ha="right", fontsize=8)
        ax2.set_xlabel("Marker protein index")
        ax2.set_ylabel("Bulk intensity")
        ax2.set_title(f"Distribution of Marker Protein Abundance (n={n_markers})")
        ax2.grid(axis="y", alpha=0.3)
        fig2.tight_layout()

        plt.show()
        return fig1, fig2
    
    def display_latent_box_plots(self, sample = None):
        """
        Boxplot 1: For each cell type, show the latent expression of its marker proteins
                in their assigned cell type dimension vs. all other dimensions.
        Boxplot 2: Each marker protein's latent expression in its assigned cell type
                dimension across all samples.
        """

        if sample is None:
            latent = self.latent_expression_profiles
        else:
            latent = self.latent_expression_profiles[sample, :, :][np.newaxis, :, :]        
        
        marker_map = self.protein_to_cell_map     # (n_proteins,)
        n_cell_types = self.cell_proportions.shape[1]

        # ── Boxplot 1: All proteins + marker expression assigned vs other ─
        fig1, ax1 = plt.subplots(figsize=(max(6, (n_cell_types + 1) * 3), 5))

        box_data = [latent.flatten()]
        box_colors = ["#d3d3d3"]
        positions = [0]
        tick_positions = [0]
        tick_labels = ["All proteins"]

        color_assigned = "#e8a0a0"
        color_other = "#a0c4e8"

        for i in range(n_cell_types):
            marker_idx = np.where(marker_map == i)[0]
            if len(marker_idx) == 0:
                continue

            assigned_vals = latent[:, marker_idx, i].flatten()
            non_marker_idx = np.where((marker_map != i) & (marker_map != -1))[0]
            if len(non_marker_idx) == 0:
                non_marker_idx = np.where(marker_map == -1)[0]
            other_vals = latent[:, non_marker_idx, i].flatten()

            base = (i + 1) * 3
            positions.extend([base, base + 1])
            box_data.extend([assigned_vals, other_vals])
            box_colors.extend([color_assigned, color_other])
            tick_positions.append(base + 0.5)
            tick_labels.append(f"Type {i + 1}")

        bp1 = ax1.boxplot(
            box_data,
            positions=positions,
            widths=0.7,
            patch_artist=True,
            medianprops=dict(color="black", linewidth=1.5),
        )
        for patch, c in zip(bp1["boxes"], box_colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.8)

        ax1.set_xticks(tick_positions)
        ax1.set_xticklabels(tick_labels)
        ax1.set_xlabel("Cell Type")
        ax1.set_ylabel("Latent Expression")
        ax1.set_title("Marker Latent Expression: Assigned Dimension vs. Other Dimensions")

        legend_handles = [
            plt.Rectangle((0, 0), 1, 1, facecolor="#d3d3d3", alpha=0.8, label="All Proteins"),
            plt.Rectangle((0, 0), 1, 1, facecolor=color_assigned, alpha=0.8, label="Assigned"),
            plt.Rectangle((0, 0), 1, 1, facecolor=color_other, alpha=0.8, label="Non-markers"),
        ]
        ax1.legend(handles=legend_handles, loc="upper right")
        fig1.tight_layout()

        # ── Boxplot 2: Each marker in its assigned dimension across samples
        marker_indices = np.where(marker_map != -1)[0]
        n_markers = len(marker_indices)

        fig2, ax2 = plt.subplots(figsize=(max(8, n_markers * 0.8), 5))

        marker_data = [
            latent[:, p, marker_map[p]] for p in marker_indices
        ]
        marker_labels = [str(p) for p in marker_indices]

        # Color by cell type assignment
        cmap = plt.cm.get_cmap("tab10", n_cell_types)
        box_colors_2 = [cmap(marker_map[p]) for p in marker_indices]

        bp2 = ax2.boxplot(
            marker_data,
            patch_artist=True,
            medianprops=dict(color="black", linewidth=1.5),
        )
        for patch, c in zip(bp2["boxes"], box_colors_2):
            patch.set_facecolor(c)
            patch.set_alpha(0.8)

        ax2.set_xticks(range(1, n_markers + 1))
        ax2.set_xticklabels(marker_labels, rotation=45, ha="right", fontsize=8)
        ax2.set_xlabel("Marker Protein Index")
        ax2.set_ylabel("Latent Expression (Assigned Dimension)")
        ax2.set_title(f"Marker Protein Latent Expression in Assigned Cell Type (n={n_markers})")
        # One legend entry per cell type
        ct_handles = [
            plt.Rectangle((0, 0), 1, 1, facecolor=cmap(i), alpha=0.8, label=f"Type {i + 1}")
            for i in range(n_cell_types)
        ]
        ax2.legend(handles=ct_handles, loc="upper left", fontsize=8)
        ax2.grid(axis="y", alpha=0.3)
        fig2.tight_layout()

        plt.show()
        return fig1, fig2    
    
    def display_histograms(self):
        Y = self.bulk_sample
        protein_to_cell_map = self.protein_to_cell_map
        marker_mask = protein_to_cell_map >= 0

        # Plot 1: All bulk protein intensities
        fig1, ax1 = plt.subplots(figsize=(7, 5))
        ax1.hist(Y.flatten(), bins=60, color="steelblue", edgecolor="white", linewidth=0.4)
        ax1.set_title("Distribution of all bulk protein intensities")
        ax1.set_xlabel("Bulk intensity")
        ax1.set_ylabel("Count")
        fig1.tight_layout()
        plt.show()

        # Plot 2: Marker bulk protein intensities
        fig2, ax2 = plt.subplots(figsize=(7, 5))
        marker_Y = Y[marker_mask, :]
        ax2.hist(marker_Y.flatten(), bins=60, color="coral", edgecolor="white", linewidth=0.4)
        ax2.set_title("Distribution of marker bulk intensities")
        ax2.set_xlabel("Bulk intensity")
        ax2.set_ylabel("Count")
        fig2.tight_layout()
        plt.show()

        return fig1, fig2  
      
    def summary(self):
        '''
        Returns a Summary
        '''

        # --- Summary table — marker proteins ---
        protein_to_cell_map = self.protein_to_cell_map
        marker_mask = protein_to_cell_map >= 0
        Y = self.bulk_sample
        marker_Y = Y[marker_mask, :]
        flat_marker_Y = marker_Y.flatten()
        Y_flat = Y.flatten() 
        
        print("\n── All protein across all samples summary ──────────────────")
        print(f"  Count  : {len(Y_flat)}")
        print(f"  Mean   : {np.mean(Y_flat):.4f}")
        print(f"  Std    : {np.std(Y_flat):.4f}")
        print(f"  Min    : {np.min(Y_flat):.4f}")
        print(f"  25th % : {np.percentile(Y_flat, 25):.4f}")
        print(f"  Median : {np.median(Y_flat):.4f}")
        print(f"  75th % : {np.percentile(Y_flat, 75):.4f}")
        print(f"  Max    : {np.max(Y_flat):.4f}")
        print("────────────────────────────────────────────")

        print("\n── Marker protein summary ──────────────────")
        print(f"  Count  : {len(flat_marker_Y)}")
        print(f"  Mean   : {np.mean(flat_marker_Y):.4f}")
        print(f"  Std    : {np.std(flat_marker_Y):.4f}")
        print(f"  Min    : {np.min(flat_marker_Y):.4f}")
        print(f"  25th % : {np.percentile(flat_marker_Y, 25):.4f}")
        print(f"  Median : {np.median(flat_marker_Y):.4f}")
        print(f"  75th % : {np.percentile(flat_marker_Y, 75):.4f}")
        print(f"  Max    : {np.max(flat_marker_Y):.4f}")
        print("────────────────────────────────────────────")
    
    def visualize2(self, cell_type_names=None):
        Y = self.bulk_sample
        protein_to_cell_map = self.protein_to_cell_map

        C = len(self.cell_type_means)
        if cell_type_names is None:
            cell_type_names = [f"Type {i+1}" for i in range(C)]
        elif len(cell_type_names) != C:
            raise ValueError(f"cell_type_names length ({len(cell_type_names)}) must match n_cell_types ({C})")

        # --- Plot 1: Boxplot — All proteins + marker proteins grouped by cell type ---
        sns.set_style("whitegrid")
        fig1, ax1 = plt.subplots(figsize=(12, 8))

        rows = [{"Category": "All proteins", "Bulk intensity": val} for val in Y.flatten()]
        for c in range(C):
            cell_mask = protein_to_cell_map == c
            for val in Y[cell_mask, :].flatten():
                rows.append({"Category": cell_type_names[c], "Bulk intensity": val})
        df_p1 = pd.DataFrame(rows)

        group_order = ["All proteins"] + [cell_type_names[c] for c in range(C)]
        sns.boxplot(
            data=df_p1,
            x="Category",
            y="Bulk intensity",
            hue="Category",
            order=group_order,
            hue_order=group_order,
            ax=ax1,
            palette='husl',
            legend=False
        )

        # Add global median line
        global_median = np.median(Y.flatten())
        ax1.axhline(
            y=global_median,
            color='red',
            linestyle='--',
            linewidth=1.5,
            label=f'Global Median ({global_median:.2f})'
        )

        # Add per-group median horizontal lines (extending across each box)
        for i, group in enumerate(group_order):
            group_data = df_p1[df_p1["Category"] == group]["Bulk intensity"]
            median_val = group_data.median()
            ax1.hlines(
                y=median_val,
                xmin=i - 0.4,
                xmax=i + 0.4,
                color='black',
                linewidth=2,
                zorder=5
            )

        ax1.legend(loc='upper right')
        ax1.set_title("Cell Abundance Distribution: all proteins vs. markers by cell type" , fontsize = 16)
        ax1.tick_params(axis="x", rotation=15)
        fig1.tight_layout()

        plt.show()
        marker_mask = protein_to_cell_map >= 0

        # --- Plot 2: Boxplot — Each marker protein across all samples ---
        fig2, ax2 = plt.subplots(figsize=(14, 7))
        marker_indices = np.where(marker_mask)[0]

        rows = []
        for p in marker_indices:
            for val in Y[p, :]:
                rows.append({"Marker protein index": str(p), "Bulk intensity": val})
        df_p2 = pd.DataFrame(rows)

        sns.boxplot(
            data=df_p2,
            x="Marker protein index",
            y="Bulk intensity",
            hue="Marker protein index",
            ax=ax2,
            palette='Set3',
            legend=False,
        )
        #ax2.set_title(f"Distribution of Marker Protein Abundance (n={self.bulk_sample.shape[1]})", fontsize = 15)
        ax2.set_title(f"Distribution of Marker Protein Abundance (n={20})", fontsize = 15)

        if len(marker_indices) > 20:
            ax2.set_xticks([])
        else:
            ax2.tick_params(axis="x", rotation=45, labelsize=7)
        fig2.tight_layout()
        plt.show()

        # --- Plot 3: Histogram — all proteins across all samples ---
        fig3, ax3 = plt.subplots(figsize=(7, 5))
        ax3.hist(Y.flatten(), bins=60, color="steelblue", edgecolor="white", linewidth=0.4)
        ax3.set_title("Distribution of all bulk protein intensities")
        ax3.set_xlabel("Bulk intensity")
        ax3.set_ylabel("Count")
        fig3.tight_layout()
        plt.show()

        # --- Plot 4: Histogram — marker proteins across all samples ---
        fig4, ax4 = plt.subplots(figsize=(7, 5))
        marker_Y = Y[marker_mask, :]
        ax4.hist(marker_Y.flatten(), bins=60, color="coral", edgecolor="white", linewidth=0.4)
        ax4.set_title("Distribution of marker bulk intensities")
        ax4.set_xlabel("Bulk intensity")
        ax4.set_ylabel("Count")
        fig4.tight_layout()
        plt.show()


            
    
class ProteomicsSimulator: 
    """
    Simulates bulk proteomics data from a mixture of cell types.
    
    Uses a mixture of normals model where each protein's bulk intensity
    is a weighted sum of cell-type-specific normal distributions.
    """

    def __init__(self, n_proteins: int, n_cell_types: int, marker_counts_per_type: Union[List[int], np.ndarray, float], seed: int = 42):
        self.n_proteins = n_proteins
        self.n_cell_types = n_cell_types
        self.marker_counts_per_type = marker_counts_per_type
        self.seed = seed

        self.rng = None
        self.cell_type_means = None
        self.protein_cell_means = None
        self.protein_to_cell_map = None
        self.latent_expression_profiles = None
        self.cell_proportions = None
        self.bulk_sample = None

    def _generate_cell_profiles(
        self,
        rng: np.random.Generator,
        shuffle: bool = False,
        cell_baseline_mu: Union[int, float, List] = 16,
        cell_baseline_sigma: Union[int,float] =  1,
        protein_baseline_sigma: Union[int, float] = 1,
        cell_type_means:Union[List, None, np.ndarray] = None): 
    
        """
        Simulates cell-type-specific baseline expression levels and assigns marker proteins.

        This function performs two main tasks:
        1. Generates a unique mean expression baseline for each cell type.
        2. Maps specific proteins to be 'markers' for a cell type, while the remaining 
        proteins are designated as background/non-markers (-1).

        Args:
            n_proteins: Total number of proteins in the simulation.
            n_cell_types: Total number of cell types to simulate.
            marker_counts_per_type: 
                - If List/Array: The number of marker proteins assigned to each cell type 
                index (must have length equal to n_cell_types).
                - If float: A fraction (0.0 to 1.0) representing the percentage of 
                n_proteins to be used as markers for EACH cell type.
            rng: NumPy random generator for reproducibility.
            shuffle: If True, randomly shuffles the protein-to-cell mapping.
            cell_baseline_mu: The global mean used to sample cell-type-specific baselines.
            cell_baseline_sigma: The standard deviation of the cell-type-specific baselines.
            protein_baseline_sigma: The standard deviation for the protein baselines. These will be added with cell_baseline_mu to create a specfic protein,cell value

        Returns:
            tuple: (cell_type_means, protein_to_cell_map)
                - cell_type_means: Array of length n_cell_types containing baseline values.
                - protein_to_cell_map: Array of length n_proteins where each value is 
                the cell type index the protein marks, or -1 for background.
        """
        marker_counts_per_type = self.marker_counts_per_type

        # Error Handling
        if len(marker_counts_per_type) != self.n_cell_types:
            raise ValueError("marker_counts_per_type length must match n_cell_types")
        if sum(marker_counts_per_type) > self.n_proteins:
            raise ValueError("Total markers cannot exceed total number of proteins.")
        if isinstance(marker_counts_per_type, float):
            marker_counts_per_type = [int(marker_counts_per_type * self.n_proteins)] * self.n_cell_types

        if isinstance(cell_baseline_mu, (int, float)) and cell_type_means is None:
            # mean baseline for each cell type (mu_c)
            cell_type_means  = cell_baseline_mu + rng.normal(0, cell_baseline_sigma, size=(self.n_cell_types))
        elif len(cell_type_means) != self.n_cell_types:
            raise ValueError(f"cell_type_means length ({len(cell_type_means)}) must match n_cell_types ({self.n_cell_types})")
        
        cell_type_means = np.array(cell_type_means) # In case the user inputs a list instead of numpy array
        marker_protein_cell_means = cell_type_means + rng.normal(0, protein_baseline_sigma, size=(self.n_proteins, self.n_cell_types))

        # Marker Protein Logic
        protein_to_cell_map = np.full(shape = self.n_proteins, fill_value= -1)

        # Fill in the markers based on the counts provided
        current_idx = 0
        for cell_type_idx, count in enumerate(marker_counts_per_type):
            protein_to_cell_map[current_idx : current_idx + count] = cell_type_idx
            current_idx += count

        # This ensures markers are distributed randomly across the protein indices
        if shuffle:
            rng.shuffle(protein_to_cell_map)

        return cell_type_means, marker_protein_cell_means, protein_to_cell_map

    def _generate_latent_expression_profiles(self,
        n_samples, protein_cell_means, 
        marker_assignments, rng, o_noise, non_marker_mean, non_marker_std=1, 
        marker_std=3):

        '''
        Generates latent expression profiles for each sample

        '''
        if protein_cell_means is None:
            raise ValueError("protein_cell_means was not initialized properly")
        if protein_cell_means.shape != (self.n_proteins, self.n_cell_types):
            raise ValueError(f"protein_cell_means size {protein_cell_means.shape} must match tuple ({(self.n_proteins, self.n_cell_types)}")
        if len(marker_assignments) != self.n_proteins:
            raise ValueError(f"marker_assignments length must be {self.n_proteins}.")

        latent_profiles = np.zeros((n_samples, self.n_proteins, self.n_cell_types))
        
        marker_assignments = np.reshape(marker_assignments, (self.n_proteins, 1)) # converts (n_proteins,) to (n_proteins, 1)
        mask = marker_assignments[None, :] == np.arange(self.n_cell_types) # mask = (1 (samples), n_proteins, n_cell_types) = (1, n_proteins, 1) == (n_cell_types, )

        marker_values = (
            protein_cell_means[None, :, :] 
            + rng.normal(0, marker_std, size=(n_samples, self.n_proteins, self.n_cell_types)) 
            + rng.normal(0, o_noise, size=(n_samples, self.n_proteins, self.n_cell_types))
        ) # (n_samples, n_proteins, n_cell_types)

        non_marker_values = ( 
        rng.normal(non_marker_mean, non_marker_std, size=(n_samples, self.n_proteins, self.n_cell_types)) 
        + rng.normal(0, o_noise, size=(n_samples, self.n_proteins, self.n_cell_types)))

        latent_profiles = np.where(mask, marker_values, non_marker_values)
     
        return latent_profiles

    def _generate_cell_proportions(self, n_samples, rng, concentration = 100, ratio = None):
        '''
        Generate cell proportions for each sample using Dirichlet distribution.
        
        Args:
            n_samples: Number of samples to generate
            ratio: Target proportions for each cell type. If None, use uniform (1/n_cell_types each).
                Must sum to 1 or will be normalized.
            concentration: Controls variance around target ratio.
                        Higher = proportions closer to target (less variation)
                        Lower = proportions more variable
        '''
        if ratio is None:
            ratio = np.ones(self.n_cell_types) / self.n_cell_types
        elif len(ratio) == self.n_cell_types:
            # Normalize Ratio 
            ratio = np.array(ratio)
            ratio = ratio / ratio.sum()
        else:
            raise ValueError(f"len(ratio) must equal n_cell_types; got {len(ratio)}, expected {self.n_cell_types}")


        alpha = ratio * concentration
        pi = rng.dirichlet(alpha, n_samples)
        return pi

    def _generate_bulk_sample(self,latent_expression_profiles, cell_proportions):
        '''
        Preforms a weighted average of the latent_expression_matrix 
        '''
        pi = cell_proportions[:, np.newaxis, :]  # (n_samples, 1, n_cell_types)
        bulk = np.sum(latent_expression_profiles * pi, axis=2)  # (n_samples, n_proteins)
        return bulk.T  # (n_proteins, n_samples)
    
    def run_simulator(self, 
        n_samples: int, 
        observation_noise_std: float, 
        cell_proportions_ratio: Union[List, np.ndarray] = None, 
        concentration: int = 100,
        shuffle = False, 
        cell_baseline_mu: Union[int, float] = 16, 
        cell_baseline_sigma: Union[int, float] = 0.5,
        protein_baseline_sigma: Union[int, float] = 0.5,
        non_marker_mean: Union[int, float] = 16, 
        non_marker_std: Union[int, float] = 1, 
        marker_std: Union[int, float] = 3,
        cell_type_means: Union[List, None, np.ndarray] = None):
        '''
        Runs the simulator
        observation_noise_std = the variance of the noise that will be present when generating the latent_expression_profiles (uses Normal(0, o_noise))
        '''
    
        rng = np.random.default_rng(seed=self.seed)

        cell_type_means, protein_cell_means, protein_to_cell_map = self._generate_cell_profiles(
            rng = rng,
            shuffle = shuffle, 
            cell_baseline_mu = cell_baseline_mu,
            cell_baseline_sigma = cell_baseline_sigma,
            protein_baseline_sigma = protein_baseline_sigma,
            cell_type_means = cell_type_means)

        latent_expression_profiles = self._generate_latent_expression_profiles(
            n_samples = n_samples, 
            protein_cell_means = protein_cell_means, 
            marker_assignments = protein_to_cell_map,
            rng = rng,
            o_noise = observation_noise_std, 
            non_marker_mean = non_marker_mean, 
            non_marker_std = non_marker_std, 
            marker_std = marker_std)

        cell_proportions = self._generate_cell_proportions(
            n_samples = n_samples,
            concentration = concentration, 
            rng = rng,
            ratio = cell_proportions_ratio)  
        
        bulk_sample = self._generate_bulk_sample(
            latent_expression_profiles=latent_expression_profiles, 
            cell_proportions=cell_proportions)
        
        #Saving all info to the class
        self.rng = rng
        self.cell_type_means = cell_type_means
        self.protein_cell_means = protein_cell_means
        self.protein_to_cell_map = protein_to_cell_map
        self.latent_expression_profiles = latent_expression_profiles
        self.cell_proportions = cell_proportions
        self.bulk_sample = bulk_sample

        return SimulationResults(
            bulk_sample = self.bulk_sample,
            latent_expression_profiles = self.latent_expression_profiles,
            cell_proportions = self.cell_proportions,
            protein_to_cell_map = self.protein_to_cell_map,
            cell_type_means = self.cell_type_means,
            protein_cell_means = protein_cell_means
        )

    def rmse(self, y_pred, y_true = None, reduce_mean = False):
        """
            Calculates Root Mean Squared Error.
            
            Args:
                y_true: Ground truth array.
                y_pred: Predicted array.
                reduce_mean: If True, returns the average RMSE across all samples. 
                            If False, returns an array of RMSE values (one per row).
            """
        
        if y_true == None:
            y_true = self.cell_proportions
        y_true = np.array(y_true) 
        y_pred = np.array(y_pred)

        if y_true.shape != y_pred.shape:
            raise ValueError(f"Shape of y_true and y_pred must be equal; y_true.shape = {y_true.shape}, y_pred.shape = {y_pred.shape}")
        
        value = np.sqrt(np.mean(np.square(y_true - y_pred), axis=-1))

        if reduce_mean:
            return np.mean(value)
            
        return value
    
