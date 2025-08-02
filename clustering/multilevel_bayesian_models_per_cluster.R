library(ggplot2)
library(dplyr)
library(brms)
options(brms.backend = "cmdstanr")
library(rstan)
library(posterior)

interaction_data <- read.csv("../data/mfpca_interaction_data.csv")
cluster_data <- read.csv("../data/MFPCA_scores_with_clusters.csv")

# Join cluster labels into the main data
interaction_data <- interaction_data %>%
  mutate(PID = as.integer(PID)) %>%
  left_join(cluster_data %>% mutate(PID = as.integer(PID)), by = "PID")

summary(model_cluster1)
summary(model_cluster2)

y <- interaction_data$Expect

# Prepare data with transformed trust
interaction_data <- interaction_data %>%
  mutate(
    y_transformed = max(y, na.rm = TRUE) + 1 - y,
    Cluster = as.factor(Cluster)
  )

# Replace 0 values with 1 in Gaze_Duration, convert Cluster to factor
interaction_data <- interaction_data %>%
  mutate(
    Gaze_Duration_clean = ifelse(GazeDuration <= 0, 1, GazeDuration),
    Cluster = as.factor(Cluster)
  )

# Fit Gamma(log) model on y
model_joint <- brm(
  y_transformed ~ s(Interaction_No, by = Cluster) + Cluster + (1 + Interaction_No | PID),
  data = interaction_data,
  family = Gamma(link = "log"),
  chains = 4, cores = 4, iter = 4000,
  control = list(adapt_delta = 0.95)
)

hypothesis(model_joint, "Interaction_No:Cluster2 = 0")
conditional_effects(model_joint, "Interaction_No:Cluster")

library(bayestestR)
rope(model_joint, parameter = "b_Interaction_No:Cluster2", range = c(-0.01, 0.01))

draws <- as_draws_df(model_joint)

# Filter the parameter of interest
draws_filtered <- draws %>% 
  select(`b_Interaction_No:Cluster2`) %>%
  rename(draw = `b_Interaction_No:Cluster2`)

ggplot(draws_filtered, aes(x = draw)) +
  geom_density(fill = "skyblue", alpha = 0.5) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "red") +
  labs(
    title = "Posterior Distribution of Slope Difference: Interaction_No × Cluster2",
    x = "Posterior Draws for Interaction Effect",
    y = "Density"
  ) +
  theme_minimal()

