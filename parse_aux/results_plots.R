# Written by Dánnell Quesada-Chacón

# Load Packages (some might not be necessary anymore)

library(raster)
library(transformeR)
library(visualizeR)
library(climate4R.value)
library(magrittr)
library(gridExtra)
library(RColorBrewer)
library(sp)
library(sf)
library(tidyverse)
library(ggsci)
library(cowplot)
library(matrixStats)
library(rowr)
library(ggpubr)
library(grid)
library(scales)
library(viridis)
library(ggnewscale)
library(colorspace)

# Remove warnings
options(warn = -1)

source("aux_results.R") # Functions to parse and plot

# Set path where the individual folders with the validation results are, change accordingly
path <- paste0(getwd(), "/../val_hist/")
setwd(path)

# Create folders for figures
dir.create("./boxplots")
dir.create("./spatial")
dir.create("./loss_plots")

# Load the metrics for all seeds with 1 run first, the name should match the folder created with the validation results (without the "V-")
models_1run <- raw_read("full_d-1") #

# Which metrics should be ranked ascending and descending
desc_rank <- colnames(models_1run) %>% str_subset("median") %>% str_subset("ROCSS|Spear")
asc_rank <- colnames(models_1run) %>% str_subset("median") %>% setdiff(desc_rank) 

# Filtering conditions
cond2filter <- list(RB=3, RB98S=10, RAAC=10, WAMS=1, DAMS=1.5)

# Filter NAs and rank all models
ranks_all <- models_1run %>%
    drop_na() %>%
    mutate(across(all_of(desc_rank), .fns = list(rank= ~percent_rank(desc(.x))))) %>%
    mutate(across(all_of(asc_rank), .fns = list(rank= ~percent_rank(abs(.x))))) %>%
    mutate(Sums = rowSums(across(contains("rank")))) %>%
    arrange(., Sums) %>%
    mutate(Rank = 1:nrow(.)) %>%
    as.tibble()

# Info about the failing models, to be seen in the terminal
models_na <- models_1run %>%
    filter(is.nan(median.ROCSS))%>%
    mutate(models=str_remove_all(models,"-0.25|-le_0.3-le_0.3|_0.3|RUE|ALSE"), 
           .keep = "unused", .before= iter) %>% 
    mutate(models=str_remove_all(models,"T-"), 
           .keep = "unused", .before= iter) %>%
    separate(col = models, sep = "-", into = c("Arch", "Layers", "Channels", "FLast", "BN")) %>%
    select(-contains(c("median", "Q90", "_rank", "models", "loss"))) %>%
    mutate(across(where(is_character),as_factor))

# Info used in the discussion, different grouping variables
models_na %>% group_by(seed) %>% tally
models_na %>% group_by(BN) %>% tally
models_na %>% group_by(Arch) %>% tally
models_na %>% group_by(Channels) %>% tally
models_na %>% group_by(FLast) %>% tally
models_na %>% group_by(Channels,FLast, .drop = FALSE) %>% tally

models_na %>% group_by(Channels, FLast, .drop = FALSE) %>% tally
models_na %>% group_by(Channels, BN, .drop = FALSE) %>% tally

models_na %>% group_by(BN,FLast, .drop = FALSE) %>% tally
models_na %>% group_by(Arch, Layers, FLast, .drop = FALSE) %>% tally
models_na %>% group_by(Channels, FLast, BN, .drop = FALSE) %>% tally

# Tables in appendices
models_na %>% group_by(Layers, Channels, FLast, .drop = FALSE) %>% tally %>%
    ungroup() %>%
    mutate(Layers = fct_relevel(Layers, c("3", "4", "5")),
           FLast = fct_relevel(FLast, c("1", "3"))) %>%
    relocate(FLast, .after = Layers) %>%
    arrange(Layers, FLast, Channels) %>% 
    write.csv("models_na_LCF_det.csv") # write csv if desired

models_na %>% group_by(Layers, FLast, BN, .drop = FALSE) %>% tally %>%
    ungroup() %>%
    mutate(Layers = fct_relevel(Layers, c("3", "4", "5")),
           FLast = fct_relevel(FLast, c("1", "3")),
           BN = fct_relevel(BN, c("T", "F"))) %>%
    relocate(FLast, .after = Layers) %>%
    arrange(Layers, FLast, BN) %>% 
    write.csv("models_na_LCN_det.csv") # write csv if desired

# Apply conditions to general ranking
df_C1_all <- ranks_all %>%
    filter(median.RB >= -cond2filter$RB, median.RB < cond2filter$RB,
           median.RB98Sto >= -cond2filter$RB98S, median.RB98Sto < cond2filter$RB98S,
           median.RAAC >= -cond2filter$RAAC, median.RAAC <= cond2filter$RAAC,
           median.WetAMS >= -cond2filter$WAMS, median.WetAMS <= cond2filter$WAMS,
           median.DryAMS >= -cond2filter$DAMS, median.DryAMS <= cond2filter$DAMS) %>%
    mutate(Rank2 = 1:nrow(.)) 

# Remove duplicated models, to show other ones
df_C1_all %<>% distinct(models, .keep_all = TRUE)

# Amount of models to include in first plots
nm2plot <- 12

# Plot in the discussion, conditions applied
plot_C1_all <- bind_rows(ranks_all %>% filter(models=="CNN1") %>% .[1,],
                         df_C1_all[1:(nm2plot-1),]) %>% 
    filter(if_any(everything(), ~ !is.na(.)))

# Figure 3 in the manuscript
df_filter_plot(plot_C1_all, "f03")

# Figure 4 in the manuscript, 1:7 are the models from plot_C1_all and  c(1,3,4,5,6,7) the metrics to plot
spatial_plot(plot_C1_all, c(1:7), c(1,3,4,5,6,7), "f04")

# Plot without conditions in the appendices, af01
plot_all <- bind_rows(ranks_all %>% filter(models=="CNN1") %>% .[1,],
                      ranks_all[1:(nm2plot-1),])
                      
df_filter_plot(plot_all, "af01")

# Reproducibility plots

# For figure 5 of the manuscript, note that 1 and 1-2 (also 2 -- 2-2 and 3 -- 3-2) 
# are the same but different names to avoid overwriting, 1 with same GPU, 2 and 3 with different ones
models_f05 <- raw_read(c("1_d-0","1_d-1","1-2_d-1",
                         "2_d-0","2_d-1","2-2_d-1",
                         "3_d-0","3_d-1","3-2_d-1"))

# To check if they are exactly the same
models_f05 %>% group_by(models,iter) %>% 
    summarise(sd = sd(median.ROCSS))

models2plot <- models_f05 %>% 
    select(models, seed, iter) %>%
    unique %>% 
    mutate(label = case_when(str_detect(as.character(iter), "d-0") ~ "ND",
                             str_detect(as.character(iter), "-2") ~ "D #2",
                             TRUE ~ "D #1"))

repro_plot_2(reprodf_runs(models2plot), reprodf_metrics(models2plot), 
             "f05", breaks_epochs = 3)

# repro_plot(reprodf_runs(models2plot), reprodf_metrics(models2plot),
#            "f05_01", breaks_epochs = 3)

# For figure 6 of the manuscript

models_f06 <- raw_read(c("1_d-0", "5_d-0", "4_d-0"))

models_f06_rev <- models_f06 %>%
    select(models, seed, iter) %>%
    distinct(models, seed, .keep_all = TRUE) %>%
    mutate(label = "ND") %>% 
    filter(!(models=="CNN1" & seed!="4096")) %>% # CNN1 with 30889 is removed to show the other one
    filter(!(str_detect(models, "U-3-64"))) %>% # U-3-64 is removed because it was shown before
    filter(!(str_detect(models, "Upp-4-128"))) %>%
    .[-nrow(.),] # Remove the last one because some names are not properly shown within the plot, due to its width

repro_plot(reprodf_runs(models_f06_rev), 
           reprodf_metrics(models_f06_rev), 
           "f06", breaks_epochs = 3)

# For figure a02
models_loss_1 <- raw_read(c("1_d-0", "5_d-0", "2_d-0", "3_d-0"))
models_loss_2 <- raw_read("4_d-0")

models2loss_1 <- models_loss_1 %>% 
    select(models, seed, iter) %>%
    distinct(models, seed, .keep_all = TRUE)

models2loss_2 <- models_loss_2 %>% 
    select(models, seed, iter) %>%
    distinct(models, seed, .keep_all = TRUE) %>%
    filter(str_detect(models, "U-4-128|Upp-4-64|Upp-4-128"))

models2loss <- rbind.data.frame(models2loss_1, models2loss_2)
    
train_loss_plot(models2loss, "af02")
