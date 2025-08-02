# load package
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

oopts <- options(max.print = 24, digits = 4, scipen = 1)
options(oopts)

interaction_data <- read.csv("../data/mfpca_interaction_data.csv")
head(interaction_data)
glimpse(interaction_data)
unique(interaction_data$Interaction_No)  
length(unique(interaction_data$PID))


make_funData <- function(interaction_data, variable) {
  argvals <- 1:32  # since you know time points go from 1 to 32
  
  interaction_data_wide <- interaction_data %>%
    select(PID, Interaction_No, !!sym(variable)) %>%
    mutate(Interaction_No = as.character(Interaction_No)) %>%
    pivot_wider(
      names_from = Interaction_No,
      values_from = all_of(variable),
      values_fill = NA  # fill missing interaction numbers with NA
    ) %>%
    arrange(PID)
  
  # Make sure all columns from 1 to 32 exist and are in order
  interaction_data_wide <- interaction_data_wide %>%
    select(PID, all_of(as.character(argvals)))
  
  X <- as.matrix(interaction_data_wide[, -1, drop = FALSE])
  
  fd <- funData(argvals = argvals, X = X)
  names(fd) <- paste0("PID_", interaction_data_wide$PID)
  return(fd)
}

object_vars <- c("mean_dist", "Gaze_on_AGV", "User_Relative_Speed", "Frechet_Distance")
object_vars

objdata <- multiFunData(
  setNames(
    lapply(object_vars, function(v) make_funData(interaction_data, v)),
    object_vars
  )
)

var_names <- c("mean_dist", "Gaze_on_AGV", "User_Relative_Speed", "Frechet_Distance")
# Ensure PID is a factor or numeric
interaction_data$PID <- as.factor(interaction_data$PID)

# Get every 10th PID
pids <- sort(unique(interaction_data$PID))
pids_subset <- pids[seq(1, length(pids), by = 10)]

# Match funData row names
fun_names <- names(objdata[[1]])
fun_pids <- as.numeric(gsub("PID_", "", fun_names))

# Find matching row indices
keep_indices <- which(fun_pids %in% as.numeric(pids_subset))

# Ensure match in lengths
stopifnot(length(var_names) == length(objdata))

## nObs
nObs(objdata)

## nObsPoints
nObsPoints(objdata)


# Example: Assign 5 PCs to first 2 variables, 4 PCs to the rest
uniExpansions <- lapply(seq_along(objdata), function(i) {
  npc_val <- if (i <= 2) 5 else 4
  list(type = "uFPCA", npc = npc_val)
})


n_pcs <- 4        
n_clusters <- 2    

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
obs <- c(1, 5, 10)  # or any PID row indices

# --- Scree plots
screeplot(mfpca_objdata, main = "Screeplot - lines")
screeplot(mfpca_objdata, type = "barplot", main = "Screeplot - barplot")

# --- Score plot
scoreplot(mfpca_objdata, cex = 0.8, main = "MFPCA Scoreplot", col = rgb(0, 0, 0, alpha = 0.7))

# --- Clustering based on first n_pcs
scores <- mfpca_objdata$scores
scores_to_use <- scores[, 1:n_pcs]
colnames(scores_to_use) <- paste0("PC", 1:n_pcs)
rownames(scores_to_use) <- rownames(scores)

# K-means clustering
set.seed(123)
wss <- sapply(1:10, function(k) {
  kmeans(scores_to_use, centers = k, nstart = 20)$tot.withinss
})


fviz_nbclust(scores_to_use, kmeans, method = "silhouette", k.max = 10) +
  labs(title = "Silhouette Method for Optimal k")

# --- Run k-means clustering
kmeans_result <- kmeans(scores_to_use, centers = n_clusters)

# --- Prepare scores data frame with cleaned PIDs
scores_df <- as.data.frame(scores_to_use)
colnames(scores_df) <- paste0("PC", seq_len(ncol(scores_to_use)))

# Remove "PID_" prefix and convert to numeric
scores_df$PID <- as.numeric(gsub("PID_", "", rownames(scores_to_use)))

# Add cluster assignments
scores_df$Cluster <- as.factor(kmeans_result$cluster)

# --- Scatter plot with PID labels
ggplot(scores_df, aes(x = PC1, y = PC2, color = Cluster, label = PID)) +
  geom_point(size = 3) +
  geom_text(vjust = -0.5, size = 3) +
  labs(
    title = "Clusters Based on MFPCA Scores",
    x = "MFPCA PC1",
    y = "MFPCA PC2"
  ) +
  theme_minimal()

# --- Save results
saveRDS(kmeans_result, "kmeans_result.rds")
write.csv(scores_df, file = "../data/MFPCA_scores_with_clusters_DV.csv", row.names = FALSE)


