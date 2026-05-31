library("funData")
library("MFPCA")
library("ggplot2")
library("refund")
library("fda")
library("dplyr")
library("tidyr")
library("tidyverse")
library("gridExtra")
library("factoextra")
library("scales")
library("trajr")
library("purrr")
library("cluster")
library("dendextend")
library("here")

oopts <- options(max.print = 24, digits = 4, scipen = 1)
options(oopts)

# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------

input_path <- "C:/Users/HIALAB/Box/Human_AGV_project/Modeling/human-agv-interaction/data/processed/behavioral_clustering_data.csv"
per_second_data <- read.csv(input_path)

head(per_second_data)
glimpse(per_second_data)
length(unique(per_second_data$PID))

per_second_data$PID <- as.factor(per_second_data$PID)

# ------------------------------------------------------------
# Convert each behavioral variable to funData
# ------------------------------------------------------------

make_funData <- function(per_second_data, variable) {
  argvals <- sort(unique(per_second_data$Time_Index))
  
  per_second_data_wide <- per_second_data %>%
    dplyr::select(PID, Time_Index, !!sym(variable)) %>%
    pivot_wider(
      names_from = Time_Index,
      values_from = all_of(variable),
      values_fill = NA
    ) %>%
    arrange(PID)
  
  X <- as.matrix(per_second_data_wide[, -1])
  fd <- funData(argvals = argvals, X = X)
  names(fd) <- paste0("PID_", per_second_data_wide$PID)
  
  return(fd)
}

object_vars <- c(
  "Frechet_Distance",
  "Gaze_Angle_to_AGV",
  "User_Speed",
  "Gaze_Instability"
)

var_names <- object_vars

objdata <- multiFunData(
  setNames(
    lapply(object_vars, function(v) make_funData(per_second_data, v)),
    object_vars
  )
)

nObs(objdata)
nObsPoints(objdata)

# ------------------------------------------------------------
# Choose number of PCs for MFPCA
# ------------------------------------------------------------

# For k = 3 or 5, I recommend starting with 3 PCs.
# You can later try 2, 3, 4 and compare silhouette results.
n_pcs <- 3

# ------------------------------------------------------------
# Run MFPCA
# ------------------------------------------------------------

uniExpansions <- lapply(seq_along(objdata), function(i) {
  list(type = "uFPCA", npc = n_pcs)
})

mfpca_objdata <- MFPCA(
  objdata,
  M = n_pcs,
  uniExpansions = uniExpansions,
  fit = TRUE
)

summary(mfpca_objdata)

# ------------------------------------------------------------
# Plot original curves and fitted MFPCA curves
# ------------------------------------------------------------

fun_names <- names(objdata[[1]])
obs <- which(names(objdata[[1]]) %in% fun_names[seq(1, length(fun_names), by = 25)])

plots <- list()

for (i in seq_along(objdata)) {
  color_vec <- hue_pal()(length(obs))
  
  g <- autoplot(objdata[[i]], obs = obs) +
    labs(
      x = "Normalized Time",
      y = var_names[i],
      col = "PID"
    ) +
    scale_y_continuous(
      limits = range(
        objdata[[i]]@X[obs, ],
        mfpca_objdata$fit[[i]]@X[obs, ],
        na.rm = TRUE
      )
    ) +
    geom_line(aes(colour = obs)) +
    autolayer(
      mfpca_objdata$fit[[i]],
      obs = obs,
      lty = 2,
      col = rep(color_vec, each = nObsPoints(mfpca_objdata$fit[[i]]))
    ) +
    theme_bw(base_size = 12)
  
  plots[[i]] <- g
}

gridExtra::grid.arrange(
  grobs = plots,
  nrow = 1
)

# ------------------------------------------------------------
# Scree plots
# ------------------------------------------------------------

screeplot(mfpca_objdata, main = "Screeplot - lines")
screeplot(mfpca_objdata, type = "barplot", main = "Screeplot - barplot")

# ------------------------------------------------------------
# Mean function plots
# ------------------------------------------------------------

g_mean <- autoplot(mfpca_objdata$meanFunction, lwd = 1.5)

g_mean_list <- lapply(seq_along(objdata), function(i) {
  g_mean[[i]] +
    scale_y_continuous(limits = range(objdata[[i]]@X, na.rm = TRUE)) +
    labs(x = "Normalized Time", y = names(objdata)[i])
})

grid.arrange(grobs = g_mean_list, ncol = 4)

# ------------------------------------------------------------
# Principal component loading plots
# ------------------------------------------------------------

pc_colors <- hue_pal()(n_pcs)

plot_var_with_color <- function(index, title) {
  autoplot(mfpca_objdata$functions[[index]]) +
    geom_hline(yintercept = 0, col = "grey70", linetype = "dashed") +
    geom_line(aes(colour = obs), linewidth = 1.25) +
    labs(title = title, x = "Normalized Time", y = "PC Loading") +
    scale_color_manual(values = pc_colors[1:n_pcs], name = "Princ. Comp.") +
    theme_bw(base_size = 12)
}

g_list <- lapply(seq_along(objdata), function(i) {
  plot_var_with_color(i, var_names[i])
})

get_legend <- function(myggplot) {
  tmp <- ggplotGrob(myggplot)
  leg <- which(sapply(tmp$grobs, function(x) x$name) == "guide-box")
  tmp$grobs[[leg]]
}

shared_legend <- get_legend(g_list[[1]])

grid.arrange(
  grobs = c(
    lapply(g_list, function(g) g + theme(legend.position = "none")),
    list(shared_legend)
  ),
  ncol = length(g_list) + 1,
  widths = c(rep(1, length(g_list)), 0.3)
)

# ------------------------------------------------------------
# Score plot
# ------------------------------------------------------------

scoreplot(
  mfpca_objdata,
  cex = 0.8,
  main = "MFPCA Scoreplot",
  col = rgb(0, 0, 0, alpha = 0.7)
)

# ------------------------------------------------------------
# Prepare scores for clustering
# ------------------------------------------------------------

scores <- mfpca_objdata$scores
scores_to_use <- scores[, 1:n_pcs, drop = FALSE]

colnames(scores_to_use) <- paste0("PC", 1:n_pcs)
rownames(scores_to_use) <- rownames(scores)

# ------------------------------------------------------------
# Evaluate possible number of clusters
# ------------------------------------------------------------

kmax <- 10
set.seed(123)

dist_mat <- dist(scores_to_use, method = "euclidean")

wss <- sapply(1:kmax, function(k) {
  kmeans(scores_to_use, centers = k, nstart = 25)$tot.withinss
})

sil_kmeans <- sapply(2:kmax, function(k) {
  cl <- kmeans(scores_to_use, centers = k, nstart = 25)$cluster
  mean(silhouette(cl, dist_mat)[, 3])
})

kmeans_scores <- data.frame(
  k = 1:kmax,
  WSS = wss,
  Silhouette = c(NA, sil_kmeans)
)

print(kmeans_scores)

fviz_nbclust(scores_to_use, kmeans, method = "silhouette", k.max = kmax) +
  labs(title = paste0("Silhouette Method for Optimal k using ", n_pcs, " PCs"))

fviz_nbclust(scores_to_use, kmeans, method = "wss", k.max = kmax) +
  labs(title = paste0("Elbow Method for Optimal k using ", n_pcs, " PCs"))

# ------------------------------------------------------------
# Hierarchical clustering evaluation
# ------------------------------------------------------------

hc <- hclust(dist_mat, method = "ward.D2")

ks <- 2:min(kmax, attr(dist_mat, "Size"))

sil_hc <- sapply(ks, function(k) {
  cl <- cutree(hc, k = k)
  sil <- silhouette(cl, dist_mat)
  mean(sil[, "sil_width"], na.rm = TRUE)
})

hier_scores <- data.frame(
  k = ks,
  Silhouette = as.numeric(sil_hc)
)

print(hier_scores)

# ------------------------------------------------------------
# Run clustering for k = 2, 3, and 5
# ------------------------------------------------------------

cluster_ks <- c(2, 3, 5)

# Choose method to save as main cluster assignment
# Options: "kmeans" or "hierarchical"
method <- "kmeans"

output_dir <- dirname(input_path)

all_cluster_results <- list()

plot_pc_pair <- function(scores_df, x_pc = "PC1", y_pc = "PC2") {
  ggplot(scores_df, aes_string(x = x_pc, y = y_pc, color = "Cluster", label = "PID")) +
    geom_point(size = 3) +
    geom_text(vjust = -0.5, size = 3) +
    labs(
      x = x_pc,
      y = y_pc,
      color = "Cluster"
    ) +
    theme_minimal(base_size = 12)
}

for (n_clusters in cluster_ks) {
  
  set.seed(123)
  
  # -------------------------
  # K-means
  # -------------------------
  kmeans_result <- kmeans(
    scores_to_use,
    centers = n_clusters,
    nstart = 25
  )
  
  # -------------------------
  # Hierarchical
  # -------------------------
  hc_clusters <- cutree(hc, k = n_clusters)
  
  selected_clusters <- switch(
    method,
    "kmeans" = kmeans_result$cluster,
    "hierarchical" = hc_clusters,
    stop("method must be either 'kmeans' or 'hierarchical'")
  )
  
  # -------------------------
  # Prepare output dataframe
  # -------------------------
  scores_df <- as.data.frame(scores_to_use)
  scores_df$PID <- as.numeric(gsub("PID_", "", rownames(scores_to_use)))
  scores_df$Cluster <- as.factor(selected_clusters)
  
  # Optional: also include both clustering results
  scores_df$KMeans_Cluster <- as.factor(kmeans_result$cluster)
  scores_df$Hierarchical_Cluster <- as.factor(hc_clusters)
  
  # -------------------------
  # Print comparison table
  # -------------------------
  cat("\n====================================\n")
  cat("Number of clusters:", n_clusters, "\n")
  cat("Number of PCs:", n_pcs, "\n")
  cat("Selected method:", method, "\n")
  cat("====================================\n")
  
  print(table(KMeans = kmeans_result$cluster, Hierarchical = hc_clusters))
  
  # -------------------------
  # Plot PC1 vs PC2
  # -------------------------
  g_pc12 <- plot_pc_pair(scores_df, "PC1", "PC2") +
    labs(
      title = paste0(
        "Behavioral clustering: ",
        method,
        ", k = ",
        n_clusters,
        ", PCs = ",
        n_pcs
      )
    )
  
  print(g_pc12)
  
  # -------------------------
  # Plot dendrogram for this k
  # -------------------------
  dend <- as.dendrogram(hc)
  dend <- color_branches(dend, k = n_clusters)
  dend <- set(dend, "labels_cex", 0.8)
  
  plot(
    dend,
    main = paste0("Colored Dendrogram: k = ", n_clusters, ", PCs = ", n_pcs)
  )
  
  # -------------------------
  # Save CSV
  # -------------------------
  out_csv <- file.path(
    output_dir,
    paste0(
      "cluster_assignments_behavioral_",
      method,
      "_k",
      n_clusters,
      "_pc",
      n_pcs,
      ".csv"
    )
  )
  
  write.csv(scores_df, file = out_csv, row.names = FALSE)
  
  # -------------------------
  # Save RDS
  # -------------------------
  out_rds <- file.path(
    output_dir,
    paste0(
      "behavioral_",
      method,
      "_result_k",
      n_clusters,
      "_pc",
      n_pcs,
      ".rds"
    )
  )
  
  saveRDS(
    list(
      method = method,
      n_clusters = n_clusters,
      n_pcs = n_pcs,
      scores_to_use = scores_to_use,
      scores_df = scores_df,
      kmeans_model = kmeans_result,
      hierarchical_model = hc,
      selected_clusters = selected_clusters,
      kmeans_clusters = kmeans_result$cluster,
      hierarchical_clusters = hc_clusters
    ),
    out_rds
  )
  
  all_cluster_results[[paste0("k", n_clusters)]] <- scores_df
}

# ------------------------------------------------------------
# Save summary of k-selection metrics
# ------------------------------------------------------------

write.csv(
  kmeans_scores,
  file = file.path(output_dir, paste0("behavioral_kmeans_k_selection_pc", n_pcs, ".csv")),
  row.names = FALSE
)

write.csv(
  hier_scores,
  file = file.path(output_dir, paste0("behavioral_hierarchical_k_selection_pc", n_pcs, ".csv")),
  row.names = FALSE
)
