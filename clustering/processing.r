install.packages("trajr")
library(dplyr)
library(SimilarityMeasures)
library(trajr)


study_data <- read.csv("../data/study_data_by_sec.csv")
head(study_data)
glimpse(study_data)

study_data <- study_data %>%
  group_by(PID, DRate, AGVname) %>%
  mutate(
    # Create lagged columns
    lag_User_X = lag(User_X),
    lag_User_Y = lag(User_Y),
    lag_Expected_User_X = lag(Expected_User_X),
    lag_Expected_User_Y = lag(Expected_User_Y),
    
    # Compute Frechet Distance per step
    Frechet_Distance = ifelse(
      row_number() == 1, NA_real_,
      mapply(function(curr_ux, curr_uy, curr_ex, curr_ey,
                      prev_ux, prev_uy, prev_ex, prev_ey) {
        if (any(is.na(c(curr_ux, curr_uy, curr_ex, curr_ey,
                        prev_ux, prev_uy, prev_ex, prev_ey)))) {
          return(NA_real_)
        }
        actual_traj <- matrix(c(prev_ux, prev_uy, curr_ux, curr_uy), 
                              ncol = 2, byrow = TRUE)
        expected_traj <- matrix(c(prev_ex, prev_ey, curr_ex, curr_ey), 
                                ncol = 2, byrow = TRUE)
        Frechet(actual_traj, expected_traj)
      },
      curr_ux = User_X, curr_uy = User_Y,
      curr_ex = Expected_User_X, curr_ey = Expected_User_Y,
      prev_ux = lag_User_X, prev_uy = lag_User_Y,
      prev_ex = lag_Expected_User_X, prev_ey = lag_Expected_User_Y
      )
    )
  ) %>%
  ungroup() %>%
  select(-starts_with("lag_"))


study_data <- study_data %>%
  group_by(PID) %>%
  group_modify(~ {
    sub_df <- .x
    
    # If any coordinate is NA, assign NA for speed
    if (any(is.na(sub_df$User_X) | is.na(sub_df$User_Y))) {
      sub_df$User_Relative_Speed <- NA
      return(sub_df)
    }
    
    # Create trajectory (assuming 1-second intervals)
    traj_obj <- TrajFromCoords(data.frame(
      x = sub_df$User_X,
      y = sub_df$User_Y
    ), fps = 1)
    
    # Compute velocity
    v <- TrajVelocity(traj_obj, diff = "central")
    
    # Assign speed (magnitude)
    sub_df$User_Relative_Speed <- Mod(v) / 100
    
    return(sub_df)
  }) %>%
  ungroup()

study_data$Frechet_Distance[study_data$Frechet_Distance == -1] <- NA

# Print the results (or save as needed)
glimpse(study_data)

output_path <- file.path(dirname("../data/study_data_by_sec.csv"), "study_data_by_sec.csv")
write.csv(study_data, file = output_path, row.names = FALSE)

          