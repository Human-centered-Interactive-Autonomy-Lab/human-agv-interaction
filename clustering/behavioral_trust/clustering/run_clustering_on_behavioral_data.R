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
# Make sure the working directory is set to the script's location (RStudio: Session → Set Working Directory → To Source File Location
input_path <- here::here("data", "processed", "behavioral_clustering_data.csv")
per_second_data <- read.csv(input_path)
head(per_second_data)
glimpse(per_second_data)
length(unique(per_second_data$PID))

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

object_vars <- c('Frechet_Distance', 'Gaze_Angle_to_AGV', 'User_Speed', 'Gaze_Instability')

objdata <- multiFunData(
  setNames(
    lapply(object_vars, function(v) make_funData(per_second_data, v)),
    object_vars
  )
)

class(per_second_data$PID)

# Ensure PID is a factor
per_second_data$PID <- as.factor(per_second_data$PID)

# Every 10th PID
pids <- sort(unique(per_second_data$PID))
pids_subset <- pids[seq(1, length(pids), by = 15)]

# Match PIDs to funData row names
fun_names <- names(objdata[[1]])
fun_pids <- as.numeric(gsub("PID_", "", fun_names))

# Get matching row indices
keep_indices <- which(fun_pids %in% as.numeric(pids_subset))

# List to store ggplot objects
plots <- list()

var_names <- c('Frechet_Distance', 'Gaze_Angle_to_AGV', 'User_Speed', 'Gaze_Instability')

# Get every 10th PID again (as before)
pids <- sort(unique(per_second_data$PID))
pids_subset <- pids[seq(1, length(pids), by = 10)]

fun_names <- names(objdata[[1]])
fun_pids <- as.numeric(gsub("PID_", "", fun_names))
keep_indices <- which(fun_pids %in% as.numeric(pids_subset))

## nObs
nObs(objdata)

## nObsPoints
nObsPoints(objdata)

n_pcs <- 2

# --- Run MFPCA
uniExpansions <- lapply(seq_along(objdata), function(i) {
  list(type = "uFPCA", npc = n_pcs)
})

mfpca_objdata <- MFPCA(objdata,
                       M = n_pcs,
                       uniExpansions = uniExpansions,
                       fit = TRUE)
summary(mfpca_objdata)

# Choose a few user indices to visualize
obs <- c(1, 10)  # or any PID row indices

# Choose which PIDs to plot (e.g., every 20th one)
fun_names <- names(objdata[[1]])
obs <- which(names(objdata[[1]]) %in% fun_names[seq(1, length(fun_names), by = 25)])

# List to hold plots
plots <- list()

for (i in 1:3) {
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
      mfpca_objdata$fit[[i]], obs = obs, lty = 2,
      col = rep(color_vec, each = nObsPoints(mfpca_objdata$fit[[i]]))
    ) +
    theme_bw(base_size = 12)
  
  plots[[i]] <- g
}

# Arrange all three plots in one row
gridExtra::grid.arrange(
  grobs = plots,
  nrow = 1
)

# --- Scree plots
screeplot(mfpca_objdata, main = "Screeplot - lines")
screeplot(mfpca_objdata, type = "barplot", main = "Screeplot - barplot")

# --- Mean function plots per component
g_mean <- autoplot(mfpca_objdata$meanFunction, lwd = 1.5)

g_mean_list <- lapply(seq_along(objdata), function(i) {
  g_mean[[i]] + 
    scale_y_continuous(limits = range(objdata[[i]]@X, na.rm = TRUE)) +
    labs(x = "Normalized Time", y = names(objdata)[i])
})
grid.arrange(grobs = g_mean_list, ncol = 3)

# --- Parameters ---
pc_colors <- c("#B80C0C", "#0C0CB8", "#129412")

# --- Plotting function ---
plot_var_with_color <- function(index, title) {
  autoplot(mfpca_objdata$functions[[index]]) +
    geom_hline(yintercept = 0, col = "grey70", linetype = "dashed") +
    geom_line(aes(colour = obs), linewidth = 1.25) +
    labs(title = title, x = "Normalized Time", y = "PC Loading") +
    scale_color_manual(values = pc_colors[1:n_pcs], name = "Princ. Comp.") +
    theme_bw(base_size = 12)
}

# --- Create plots using var_names ---
g_list <- lapply(1:4, function(i) plot_var_with_color(i, var_names[i]))

# --- Extract shared legend from the first plot ---
get_legend <- function(myggplot) {
  tmp <- ggplotGrob(myggplot)
  leg <- which(sapply(tmp$grobs, function(x) x$name) == "guide-box")
  tmp$grobs[[leg]]
}
shared_legend <- get_legend(g_list[[1]])

# --- Arrange all three plots and shared legend ---
grid.arrange(
  grobs = c(
    lapply(g_list, function(g) g + theme(legend.position = "none")),
    list(shared_legend)
  ),
  ncol = 5,
  widths = c(1, 1, 1, 1, 0.3)
)

# --- Score plot
scoreplot(mfpca_objdata, cex = 0.8, main = "MFPCA Scoreplot", col = rgb(0, 0, 0, alpha = 0.7))

# --- Clustering based on first n_pcs
scores <- mfpca_objdata$scores
scores_to_use <- scores[, 1:n_pcs]
colnames(scores_to_use) <- paste0("PC", 1:n_pcs)
rownames(scores_to_use) <- rownames(scores)


fviz_nbclust(scores_to_use, kmeans, method = "silhouette", k.max = 5) +
  labs(title = "Silhouette Method for Optimal k")

fviz_nbclust(scores_to_use, kmeans, method = "wss", k.max = 5) +
  labs(title = "Elbow Method for Optimal k")

kmax <- 5
set.seed(123)

# Distance matrix (used for silhouette)
dist_mat <- dist(scores_to_use, method = "euclidean")

# Tot within-cluster sum of squares (Elbow)
wss <- sapply(1:kmax, function(k) {
  kmeans(scores_to_use, centers = k, nstart = 25)$tot.withinss
})

# Average silhouette width (start at k = 2)
sil <- sapply(2:kmax, function(k) {
  cl <- kmeans(scores_to_use, centers = k, nstart = 25)$cluster
  mean(silhouette(cl, dist_mat)[, 3])
})

nb_scores <- data.frame(
  k   = 1:kmax,
  WSS = wss,
  Silhouette = c(NA, sil)   # NA for k = 1 (undefined)
)
print(nb_scores)

if (!inherits(dist_mat, "dist")) dist_mat <- as.dist(dist_mat)

kmax <- 10  # set as you wish
hc <- hclust(dist_mat, method = "ward.D2")

ks <- 2:min(kmax, attr(dist_mat, "Size"))  # cannot exceed n
sil_hc <- sapply(ks, function(k) {
  cl  <- cutree(hc, k = k)
  sil <- silhouette(cl, dist_mat)
  mean(sil[, "sil_width"], na.rm = TRUE)
})

hier_scores <- data.frame(k = ks, Silhouette = as.numeric(sil_hc))
hier_scores
# K-means clustering
n_clusters <- 2
set.seed(123)
wss <- sapply(1:10, function(k) {
  kmeans(scores_to_use, centers = k, nstart = 20)$tot.withinss
})

# --- Run k-means clustering
kmeans_result <- kmeans(scores_to_use, centers = n_clusters)

# Hierarchical Clustering
dist_matrix <- dist(scores_to_use, method = "euclidean")
hc <- hclust(dist_matrix, method = "ward.D2")
plot(hc, labels = FALSE, main = "Hierarchical Clustering Dendrogram", xlab = "", sub = "")
hc_clusters <- cutree(hc, k = n_clusters)
dend_results <- data.frame(PID = rownames(scores_to_use), Cluster = hc_clusters)
dend <- as.dendrogram(hc)
dend <- color_branches(dend, k = n_clusters)
dend <- set(dend, "labels_cex", 0.8)
plot(dend, main = "Colored Dendrogram with PIDs")
dend_results$PID <- as.numeric(gsub("PID_", "", dend_results$PID))

table(kmeans_result$cluster, hc_clusters)

# Choose which clustering to use for coloring/plots/saving
method <- "kmeans"         # <- set to "kmeans" or "hierarchical"

selected_clusters <- switch(
  method,
  "kmeans"       = kmeans_result$cluster,
  "hierarchical" = hc_clusters,
  stop("method must be 'kmeans' or 'hierarchical'")
)

# --- Prepare scores data frame with cleaned PIDs ---
scores_df <- as.data.frame(scores_to_use)
colnames(scores_df) <- paste0("PC", seq_len(ncol(scores_to_use)))
scores_df$PID <- as.numeric(gsub("PID_", "", rownames(scores_to_use)))

# Use the selected clustering
scores_df$Cluster <- as.factor(selected_clusters)

# --- Plot (now uses chosen clusters) ---
plot_pc_pair <- function(x_pc, y_pc) {
  ggplot(scores_df, aes_string(x = x_pc, y = y_pc, color = "Cluster", label = "PID")) +
    geom_point(size = 3) +
    geom_text(vjust = -0.5, size = 3) +
    labs(x = x_pc, y = y_pc) +
    theme_minimal(base_size = 12) +
    theme(legend.position = "none")
}
g1 <- plot_pc_pair("PC1", "PC2")
grid.arrange(g1, ncol = 1)

# Save results
output_dir <- dirname(input_path)
if (method == "kmeans") {
  saveRDS(
    list(method = "kmeans", model = kmeans_result, clusters = selected_clusters),
    file.path(output_dir, "behavioral_kmeans_result.rds")
  )
} else {
  saveRDS(
    list(method = "hierarchical", model = hc, clusters = selected_clusters),
    file.path(output_dir, "behavioral_hierarchical_result.rds")
  )
}

out_csv <- file.path(
  output_dir,
  paste0("cluster_assignments_behavioral_", method, ".csv")
)

write.csv(scores_df, file = out_csv, row.names = FALSE)
