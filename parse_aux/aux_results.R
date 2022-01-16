# Written by Dánnell Quesada-Chacón

# Collection of functions to parse and plot the validation metrics

fix_index <- function(x, models) {
  index <- (x %>% redim(drop = TRUE))$Data
  if (length(dim(index)) == 2) {
    dim(index) <- c(1, prod(dim(index)[1:2]))
  } else {
    dim(index) <- c(nrow(index), prod(dim(index)[2:3]))
  }

  na_models <- (rowSums(is.na(index)) == ncol(index)) %>% which()
  if (length(na_models) > 0) {
    index <- index[-na_models, ]
  }
  indLand <- (!apply(index, MARGIN = 2, anyNA)) %>% which()
  index <- as.data.frame(index[, indLand] %>% t())

  if (length(na_models) > 0) {
    colnames(index) <- models[-na_models]
  } else {
    colnames(index) <- models
  }
  return(index)
}

fix_index_2 <- function(x, models) {
  index <- (x %>% redim(drop = TRUE))$Data

  dim(index) <- c(1, prod(dim(index)[1:2]))

  na_models <- (rowSums(is.na(index)) == ncol(index)) %>% which()
  if (length(na_models) > 0) {
    index <- index[-na_models, ]
  }
  indLand <- (!apply(index, MARGIN = 2, anyNA)) %>% which()
  index <- as.data.frame(index[, indLand])

  if (length(na_models) > 0) {
    colnames(index) <- models[-na_models]
  } else {
    colnames(index) <- models
  }
  return(index)
}

calc_stat <- function(x) {
  stats <- quantile(x, probs = c(0.1, 0.25, 0.5, 0.75, 0.9))
  names(stats) <- c("ymin", "lower", "middle", "upper", "ymax")
  return(stats)
}

calc_stat2 <- function(x) {
  stats <- quantile(x, probs = c(0.02, 0.25, 0.5, 0.75, 0.98))
  names(stats) <- c("ymin", "lower", "middle", "upper", "ymax")
  return(stats)
}

raw_read <- function(folders) {
  sapply(folders, simplify = F, function(u) {
    setwd(paste0(path, "/V-", u))
    seeds <- dir(path = "./", pattern = "validation", recursive = T) %>%
      str_split("/", simplify = T) %>%
      .[, 1] %>%
      unique()

    df <- sapply(seeds, simplify = F, function(x) {
      runs_CNN <- dir(path = x, pattern = "hist_train_CNN")
      validation_CNN <- dir(path = x, pattern = "validation_CNN")

      vl_tib <- sapply(1:length(runs_CNN), simplify = F, function(y) {
        print(paste0(u, "--", x, "--", y))
        load(paste0("./", x, "/", runs_CNN[y]), .GlobalEnv)

        bern_loss <- rownames_to_column(as.data.frame(t(sapply(names(history_trains), FUN = function(z) {
          m_index <- which.min(history_trains[[z]]$metrics$val_loss)

          if (length(m_index) == 1) {
            return(c(
              loss = history_trains[[z]]$metrics$loss[m_index],
              val_loss = history_trains[[z]]$metrics$val_loss[m_index],
              epochs = length(history_trains[[z]]$metrics$val_loss)
            ))
          } else {
            return(c(
              loss = NA,
              val_loss = NA,
              epochs = length(history_trains[[z]]$metrics$val_loss)
            ))
          }
        }))), var = "models")

        load(paste0("./", x, "/", validation_CNN[y]))
        ylabs <- c(
          "ROCSS", "RMSE (mm)",
          "Spearman", "RB (%)",
          "RBp98 (DET, %)", "RBp98 (STO, %)",
          "RAAC", "WetAMS (days)",
          "DryAMS (days)"
        )

        validation.list <- validation.list[-10]
        names(validation.list) <- ylabs

        for (t in grep("%)", names(validation.list))) {
          validation.list[[t]]$Data <- validation.list[[t]]$Data * 100
        }

        nicenames <- c("ROCSS", "RMSE", "Spear", "RB", "RB98Det", "RB98Sto", "RAAC", "WetAMS", "DryAMS")
        names(validation.list) <- nicenames

        if (length(names(history_trains)) == 1) {
          models_summ <- NULL
          for (v in names(validation.list)) {
            models_summ <- bind_cols(
              models_summ,
              validation.list[[v]]$Data %>%
                as.vector() %>%
                t() %>%
                as.tibble() %>%
                summarise("median.{v}" := rowMedians(as.matrix(.), na.rm = T))
            )
          }
          models_all <- bind_cols(bern_loss, models_summ)
        } else {
          models_summ <- NULL
          for (v in names(validation.list)) {
            models_summ <-
              bind_cols(
                models_summ,
                subsetDimension(validation.list[[v]],
                  dimension = "var",
                  indices = 1:length(names(history_trains))
                ) %>%
                  redim(drop = TRUE) %>%
                  .$Data %>%
                  apply(., 1L, c) %>%
                  t() %>%
                  as.tibble() %>%
                  summarise("median.{v}" := rowMedians(as.matrix(.), na.rm = T))
              )
          }
          models_all <- bind_cols(bern_loss, models_summ)
        }

        models_all$Run <- as.character(y)
        return(as.data.frame(models_all))
      }) %>% do.call(rbind.data.frame, .)

      vl_tib$seed <- as.character(x)
      return(vl_tib)
    }) %>% do.call(rbind.data.frame, .)

    df$iter <- as.character(u)
    return(df)
  }) %>%
    do.call(rbind.data.frame, .) %>%
    relocate(Run, seed, iter, .after = epochs)
}

df_filter_plot <- function(df, plot_name) {
  setwd(path)
  df_filter <- sapply(1:nrow(df), function(x) {
    load(paste0("./V-", df$iter[x], "/", df$seed[x], "/validation_CNN_", df$Run[x],".rda"))
    ylabs <- c(
      "ROCSS", "RMSE (mm)",
      "Spearman", "RB (%)",
      "RBp98D (%)", "RBp98S (%)",
      "RAAC", "WetAMS (days)",
      "DryAMS (days)"
    )

    names2 <- validation.list[[10]]
    validation.list <- validation.list[-10]
    names(validation.list) <- ylabs

    for (y in grep("%)", names(validation.list))) {
      validation.list[[y]]$Data <- validation.list[[y]]$Data * 100
    }

    validation_fix <- sapply(validation.list, fix_index, models = names2, simplify = F) %>%
      reshape2::melt() %>%
      filter(variable == df$models[x]) %>%
      droplevels()

    colnames(validation_fix) <- c("Model", "value", "metric")

    models_nice <- df$models[x] %>%
      str_remove_all("-0.25|-le_0.3-le_0.3|_0.3|RUE|ALSE") %>%
      str_remove("T-") %>%
      paste0(., ", R=", df$Rank[x], "\nS=", df$seed[x], ", E=", df$epochs[x])
    validation_fix$Model <- models_nice

    levels(validation_fix$Model) <- models_nice

    validation_fix$metric <- factor(validation_fix$metric, levels = c(
      "ROCSS", "Spearman", "RMSE (mm)",
      "RB (%)", "RBp98D (%)", "RBp98S (%)",
      "RAAC", "WetAMS (days)", "DryAMS (days)"
    ))

    return(validation_fix)
  }, simplify = F) %>% do.call(rbind.data.frame, .)

  cnn_plot <- ggplot(df_filter, aes(x = Model, y = value, color = Model)) +
    facet_wrap(~metric, scales = "free") +
    stat_summary(fun.data = calc_stat, geom = "boxplot", width = 0.6, lwd = 0.35) + # , size=0.33
    theme_light(base_size = 10, base_family = "Helvetica") +
    guides(color = guide_legend(ncol = 4)) +
    theme(
      axis.title.x = element_blank(),
      axis.text.x = element_blank(),
      axis.ticks.x = element_blank(),
      axis.title.y = element_blank(),
      strip.background = element_rect(fill = "white"),
      strip.text = element_text(color = "black", margin = margin(0, 0, 0.5, 0, unit = "mm")),
      legend.key.size = unit(2.5, "mm"),
      legend.box.margin = margin(-3.25, 8, 0, 0, unit = "mm"),
      legend.position = "bottom",
      legend.title = element_blank(),
      legend.text = element_text(margin = margin(0.2, 0.2, 0.2, 0.2, unit = "mm")),
      panel.spacing = unit(1, "mm"),
      plot.margin = margin(0, 0, -1.5, 0, unit = "mm")
    )

  if (nm2plot <= 20 && nm2plot >= 10) {
    cnn_plot <- cnn_plot + scale_color_ucscgb()
  }
  if (nm2plot <= 10) {
    cnn_plot <- cnn_plot + scale_color_jco()
  }
  if (nm2plot >= 10) {
    cnn_plot <- cnn_plot + scale_color_viridis_d(option = "turbo")
  }
  if (nm2plot == 12) {
    cnn_plot <- cnn_plot + scale_color_manual(values = c(
      "#0073c2", "#EFC000", "#A73030", "#868686",
      "#641ea4", "#76CD26", "#E67300", "#1929C8",
      "#cd2926", "#3c3c3c", "#1B6F1B", "#82491E"
    ))
  }

  ggsave(
    plot = cnn_plot, filename = paste0("./boxplots/Best-", nm2plot, "_", plot_name, ".pdf"),
    height = 100, width = 175, units = "mm"
  )

  return(paste0("Check plot in: ", path, "/boxplots"))
}

spatial_plot <- function(df, ind_selection, ind_metrics, plot_name) {
  setwd(path)
  df <- df[ind_selection, ]
  ylabs <- c(
    "ROCSS", "RMSE (mm)",
    "Spearman", "RB (%)",
    "RBp98D (%)", "RBp98S (%)",
    "RAAC", "WetAMS",
    "DryAMS"
  )

  df_s <- df %>% group_split(iter)

  validation.df <- lapply(df_s, function(m) {
    validations <- sapply(1:nrow(m), function(x) {
      load(paste0(path, "/V-", m$iter[x], "/", m$seed[x], "/validation_CNN_", m$Run[x],".rda"))
      names2 <<- validation.list[[10]]
      validation.list <- validation.list[-10]
      names(validation.list) <- ylabs

      for (y in grep("RB", names(validation.list))) {
        validation.list[[y]]$Data <- validation.list[[y]]$Data * 100
      }

      return(validation.list)
    }, simplify = F)

    names(validations) <- m$models

    models2plot <- m$models %>%
      str_remove_all("-0.25|-le_0.3-le_0.3|_0.3|RUE|ALSE") %>%
      str_remove("T-")

    models_sp <- names2 %>%
      unique() %>%
      str_remove_all("-0.25|-le_0.3-le_0.3|_0.3|RUE|ALSE") %>%
      str_remove("T-")

    ind2plot <- match(models2plot, models_sp)

    models_SR <- models2plot %>%
      paste0(., "\nR=", m$Rank, ", S=", m$seed)

    vdf <- lapply(ind_metrics, FUN = function(z) {
      lapply(1:length(validations), FUN = function(w) {
        index <- subsetDimension(validations[[w]][[z]],
          dimension = "var",
          indices = ind2plot[w]
        ) %>% redim(drop = TRUE)

        index_ext <- extent(index$xyCoords)
        rast.spdf <- raster(index$Data,
          xmn = index_ext[1], xmx = index_ext[2],
          ymn = index_ext[3], ymx = index_ext[4]
        ) %>%
          flip(direction = "y") %>%
          as(., "SpatialPixelsDataFrame") %>%
          as.data.frame() %>%
          set_colnames(c("value", "x", "y"))

        rast.spdf$metric <- as.factor(ylabs[z])
        rast.spdf$model <- models_SR[w]
        return(rast.spdf)
      }) %>% do.call(rbind.data.frame, .)
    }) %>% do.call(rbind.data.frame, .)
  }) %>%
    do.call(rbind.data.frame, .) %>%
    as.tibble()

  mod_facts <- df$models %>%
    unique() %>%
    str_remove_all("-0.25|-le_0.3-le_0.3|_0.3|RUE|ALSE") %>%
    str_remove("T-") %>%
    paste0(., "\nR=", df$Rank, ", S=", df$seed)

  validation.df %<>% mutate(model = factor(model, levels = mod_facts))

  validation.df$scales <- as.factor(ifelse(validation.df$metric %in% ylabs[c(1, 3)], "A", "B"))

  colors2plot <- colorRampPalette(colors = c("#32b732", "#b3de32", "#efef19", "#ff8c19"))(20)

  spatial_plot_nice <- ggplot() +
    geom_tile(data = validation.df %>% filter(scales == "A"), aes(x = x, y = y, fill = value)) +
    facet_grid(metric ~ model) +
    coord_sf(crs = sf::st_crs(4326)) +
    scale_x_continuous(breaks = c(13.2, 13.6, 14)) +
    scale_y_continuous(breaks = c(50.7, 50.9)) +
    theme_light(base_size = 10, base_family = "Helvetica") +
    scale_fill_gradientn(
      colours = colors2plot,
      limits = c(0.62, 0.881),
      breaks = c(0.65, 0.75, 0.85),
      name = "", labels = c(0.65, 0.75, 0.85)
    ) +
    new_scale_fill() +
    geom_tile(data = validation.df %>% filter(scales == "B"), aes(x = x, y = y, fill = value)) +
    facet_grid(metric ~ model) +
    coord_sf(crs = sf::st_crs(4326), ) +
    geom_text(
      data = validation.df %>% group_by(model, metric) %>%
        summarise(median = signif(median(value), 3)),
      aes(x = 13.8, y = 50.65, label = median), size = 2.75
    ) +
    scale_fill_gradient2(
      low = "dodgerblue2", high = "red2", mid = "gray95",
      midpoint = 0, name = "", limits = c(-43, 43), na.value = "red2"
    ) +
    theme(
      axis.title.x = element_blank(),
      axis.title.y = element_blank(),
      axis.text.x = element_text(angle = 45, hjust = 1),
      strip.background = element_rect(fill = "white"),
      strip.text = element_text(color = "black"),
      strip.text.x = element_text(margin = margin(c(0, 0, 1, 0), unit = "mm")),
      strip.text.y = element_text(margin = margin(c(0, 0, 0, 1), unit = "mm")),
      legend.key.size = unit(3, "mm"),
      legend.key.width = unit(10, "mm"),
      legend.position = "bottom",
      legend.box.margin = margin(-3.5, 0, -2, 0, unit = "mm"),
      panel.spacing = unit(1.25, "mm"),
      plot.margin = margin(0, 0, 0, 0, unit = "mm")
    )

  ggsave(paste0(path, "/spatial/Best_spatial_", plot_name, ".pdf"),
    plot = spatial_plot_nice, dpi = 600, width = 175, height = 125, units = "mm"
  )

  return(validation.df)
}

reprodf_runs <- function(df) {
  df2 <- sapply(1:nrow(df), simplify = F, function(u) {
    setwd(paste0(path, "/V-", df$iter[u]))

    seed <- as.character(df$seed[u])
    label <- as.character(df$label[u])

    runs_CNN <- dir(path = seed, pattern = "hist_train_CNN")
    validation_CNN <- dir(path = seed, pattern = "validation_CNN")

    vl_tib <- sapply(1:length(runs_CNN), simplify = F, function(y) {
      load(paste0("./", seed, "/", runs_CNN[y]), .GlobalEnv)

      bern_loss <- rownames_to_column(as.data.frame(t(sapply(as.character(df$models[u]),
        FUN = function(z) {
          m_index <- which.min(history_trains[[z]]$metrics$val_loss)

          if (length(m_index) == 1) {
            return(c(
              loss = history_trains[[z]]$metrics$loss[m_index],
              val_loss = history_trains[[z]]$metrics$val_loss[m_index],
              epochs = length(history_trains[[z]]$metrics$val_loss)
            ))
          } else {
            return(c(
              loss = NA,
              val_loss = NA,
              epochs = length(history_trains[[z]]$metrics$val_loss)
            ))
          }
        }
      ))), var = "models") # %>% arrange(., val_loss)

      bern_loss$Run <- as.character(y)
      return(as.data.frame(bern_loss))
    }) %>% do.call(rbind.data.frame, .)

    vl_tib$seed <- as.character(seed)
    vl_tib$epochs %<>% as.character()
    vl_tib$Run %<>% as.character(.) %>% factor(., levels = as.character(1:length(runs_CNN)))
    vl_tib$models %<>% unique() %>%
      str_remove_all("-0.25|-le_0.3-le_0.3|_0.3|RUE|ALSE") %>%
      str_remove("T-") %>%
      paste0(., "\nS=", seed)

    if (!identical(label, character(0))) {
      vl_tib$models %<>% paste0(., ", ", label)
    }

    levels(vl_tib$models) <- vl_tib$models %>% unique()
    vl_tib$iter <- as.character(df$iter[u])
    return(vl_tib)
  }) %>% do.call(rbind.data.frame, .)
}

reprodf_metrics <- function(df, to_use = c(1, 4, 7, 8, 9)) {
  df2 <- sapply(1:nrow(df), simplify = F, function(u) {
    setwd(paste0(path, "/V-", df$iter[u]))
    seed <- as.character(df$seed[u])
    label <- as.character(df$label[u])

    runs_CNN <- dir(path = seed, pattern = "hist_train_CNN")
    validation_CNN <- dir(path = seed, pattern = "validation_CNN")

    metric_tib <- sapply(1:length(runs_CNN), simplify = F, function(y) {
      load(paste0("./", seed, "/", runs_CNN[y]))

      load(paste0("./", seed, "/", validation_CNN[y]))
      ylabs <- c(
        "ROCSS", "RMSE",
        "Spearman", "RB",
        "RBp98D", "RBp98S",
        "RAAC", "WetAMS",
        "DryAMS"
      )

      validation.list <- validation.list[to_use]
      names(validation.list) <- ylabs[to_use]

      for (t in grep("RB", names(validation.list))) {
        validation.list[[t]]$Data <- validation.list[[t]]$Data * 100
      }

      if (length(names(history_trains)) == 1) {
        validation_fix <- sapply(validation.list, fix_index_2,
          simplify = F,
          models = names(history_trains)
        ) %>%
          reshape2::melt()
        colnames(validation_fix) <- c("Model", "value", "metric")
      } else {
        validation_fix <- sapply(validation.list, fix_index,
          simplify = F,
          models = names(history_trains)
        ) %>%
          reshape2::melt()
        colnames(validation_fix) <- c("Model", "value", "metric")
      }

      validation_fix %<>% filter(str_detect(Model, as.character(df$models[u])))

      validation_fix$metric <- factor(validation_fix$metric, levels = ylabs[to_use])
      validation_fix$Run <- as.character(y)
      return(as.data.frame(validation_fix))
    }) %>% do.call(rbind.data.frame, .)

    metric_tib$Run %<>% as.character(.) %>% factor(., levels = as.character(1:length(runs_CNN)))
    metric_tib$seed <- as.character(seed)
    metric_tib$Model %<>% unique() %>%
      str_remove_all("-0.25|-le_0.3-le_0.3|_0.3|RUE|ALSE") %>%
      str_remove("T-") %>%
      paste0(., "\nS=", seed)

    if (!identical(label, character(0))) {
      metric_tib$Model %<>% paste0(., ", ", label)
    }

    levels(metric_tib$Model) <- metric_tib$Model %>% unique()

    metric_tib$iter <- as.character(df$iter[u])
    return(metric_tib)
  }) %>% do.call(rbind.data.frame, .)
}

repro_plot <- function(runs_df, metrics_df, plot_name, breaks_epochs = 3) {
  setwd(path)
  epochs_plot <- ggplot(runs_df, aes(x = epochs, y = val_loss, color = Run, shape = Run)) +
    facet_wrap(~models, scales = "free", nrow = 1) +
    geom_point(size = 0.9, position = position_dodge(width = 0.25, preserve = "total")) +
    theme_light(base_size = 8.5, base_family = "Helvetica") +
    scale_color_manual(values = c(
      "#0073c2", "#EFC000", "#A73030", "#868686",
      "#641ea4", "#76CD26", "#E67300", "#1929C8",
      "#cd2926", "#3c3c3c"
    )) +
    scale_shape_manual(values = c(15, 1, 17, 6, 3, 18, 4, 20, 5, 0)) +
    labs(x = "Epochs", tag = "BG val loss") +
    scale_y_continuous(
      labels = scales::number_format(accuracy = 0.001),
      breaks = breaks_pretty(n = breaks_epochs)
    ) +
    guides(color = guide_legend(nrow = 1, override.aes = list(size = 1.5))) +
    theme(
      axis.title.x = element_text(margin = margin(0.5, 0, 0, 0, unit = "mm")),
      axis.text.x = element_text(angle = 90, vjust = 0.5),
      axis.text.y = element_text(angle = 55, vjust = 0.5, hjust = 0.8),
      axis.title.y = element_blank(),
      axis.title.y.right = element_text(),
      strip.background = element_rect(fill = "white"),
      strip.text = element_blank(),
      legend.position = "bottom",
      plot.tag.position = "right",
      plot.tag = element_text(angle = 270, size = 7, margin = margin(c(-14, 0.1, 0, 0.6), unit = "mm")),
      legend.text = element_text(margin = margin(0, 0.3, 0, 0.3, unit = "mm")),
      legend.key.size = unit(3, "mm"),
      legend.box.margin = margin(-3.25, 0, -1.5, 0, unit = "mm"),
      panel.spacing = unit(0, "mm"),
      plot.margin = margin(0, 0, 0, 0, unit = "mm")
    )

  cnn_plot <- ggplot(metrics_df, aes(x = Run, y = value, color = Run)) +
    facet_grid(vars(metric), vars(Model), scales = "free") +
    stat_summary(fun.data = calc_stat, geom = "boxplot", width = 0.5, lwd = 0.35) +
    theme_light(base_size = 8.5, base_family = "Helvetica") +
    scale_color_manual(values = c(
      "#0073c2", "#EFC000", "#A73030", "#868686",
      "#641ea4", "#76CD26", "#E67300", "#1929C8",
      "#cd2926", "#3c3c3c"
    )) +
    guides(color = FALSE) +
    theme(
      axis.title.x = element_blank(),
      axis.title.y = element_blank(),
      strip.background = element_rect(fill = "white"),
      strip.text = element_text(color = "black"),
      strip.text.x = element_text(margin = margin(c(0, 0, 1, 0), unit = "mm")),
      strip.text.y = element_text(margin = margin(c(0, 0, 0, 0.5), unit = "mm")),
      legend.key.size = unit(4, "mm"),
      legend.position = "bottom",
      legend.box.margin = margin(0, 0, 0, 0, unit = "mm"),
      panel.spacing = unit(1, "mm"),
      plot.margin = margin(0, 0, 1, 0, unit = "mm")
    )

  repr_plot_2 <- grid.arrange(cnn_plot, epochs_plot, heights = c(3.2, 1))
  setwd(path)
  dir.create("boxplots", recursive = TRUE)
  dir.create("spatial", recursive = TRUE)
  ggsave(
    plot = repr_plot_2, filename = paste0("./boxplots/Best_repr_", plot_name, ".pdf"),
    height = 120, width = 175, units = "mm"
  )
}

repro_plot_2 <- function(runs_df, metrics_df, plot_name, breaks_epochs = 3) {
  setwd(path)

  runs_df <- runs_df %>%
    mutate(
      Model = str_remove_all(models, ",.+$"),
      label = str_split(models, ",.", simplify = T)[, 2]
    ) %>%
    mutate(
      label = factor(label, levels = c("ND", "D #1", "D #2")),
      epochs = str_c("Epochs=", epochs)
    )

  epochs_plot <- ggplot(runs_df, aes(x = label, y = val_loss)) +
    facet_wrap(~epochs, scales = "free", nrow = 1) +
    stat_summary(fun.data = calc_stat, geom = "boxplot", width = 0.4, lwd = 0.3) +
    theme_light(base_size = 8.5, base_family = "Helvetica") +
    labs(tag = "BG val loss") +
    scale_y_continuous(
      breaks = breaks_pretty(n = breaks_epochs),
      labels = scales::number_format(accuracy = 0.001)
    ) +
    theme(
      axis.title.x = element_blank(),
      axis.text.y = element_text(angle = 55, vjust = 0.5, hjust = 0.5),
      axis.title.y = element_blank(),
      axis.title.y.right = element_text(),
      strip.background = element_rect(fill = "white"),
      strip.text = element_text(color = "black", margin = margin(0, 0, 0.5, 0, unit = "mm")),
      plot.tag.position = "right",
      plot.tag = element_text(angle = 270, size = 7, margin = margin(c(-1, 0.1, 0, 0.6), unit = "mm")),
      legend.text = element_text(margin = margin(0, 0.3, 0, 0.3, unit = "mm")),
      legend.key.size = unit(3, "mm"),
      panel.spacing = unit(1, "mm"),
      plot.margin = margin(0, 0, 0, 0, unit = "mm")
    )

  cnn_plot <- ggplot(metrics_df, aes(x = Run, y = value, color = Run)) +
    facet_grid(vars(metric), vars(Model), scales = "free") +
    stat_summary(fun.data = calc_stat, geom = "boxplot", width = 0.4, lwd = 0.25) +
    theme_light(base_size = 8.5, base_family = "Helvetica") +
    scale_color_manual(values = c(
      "#0073c2", "#EFC000", "#A73030", "#868686",
      "#641ea4", "#76CD26", "#E67300", "#1929C8",
      "#cd2926", "#3c3c3c"
    )) +
    guides(color = FALSE) +
    theme(
      axis.title.x = element_blank(),
      axis.title.y = element_blank(),
      strip.background = element_rect(fill = "white"),
      strip.text = element_text(color = "black"),
      strip.text.x = element_text(margin = margin(c(0, 0, 1, 0), unit = "mm")),
      strip.text.y = element_text(margin = margin(c(0, 0, 0, 0.5), unit = "mm")),
      legend.key.size = unit(4, "mm"),
      legend.position = "bottom",
      legend.box.margin = margin(0, 0, 0, 0, unit = "mm"),
      panel.spacing = unit(1, "mm"),
      plot.margin = margin(0, 0, 1, 0, unit = "mm")
    )

  repr_plot_2 <- grid.arrange(cnn_plot, epochs_plot, heights = c(3.35, 0.8))
  setwd(path)
  dir.create("boxplots", recursive = TRUE)
  dir.create("spatial", recursive = TRUE)

  ggsave(
    plot = repr_plot_2, filename = paste0("./boxplots/Best_repr_", plot_name, ".pdf"),
    height = 120, width = 175, units = "mm"
  )
}

train_loss_plot <- function(df, plot_name) {
  df2 <- sapply(1:nrow(df), simplify = F, function(u) {
    setwd(paste0(path, "/V-", df$iter[u]))
    seed <- as.character(df$seed[u])
    label <- as.character(df$label[u])

    runs_CNN <- dir(path = seed, pattern = "hist_train_CNN")

    train_df <- sapply(1:length(runs_CNN), simplify = F, function(y) {
      load(paste0("./", seed, "/", runs_CNN[y]), .GlobalEnv)
      tibble(
        Train = history_trains[[df$models[u]]]$metrics$loss,
        Validation = history_trains[[df$models[u]]]$metrics$val_loss,
        Seed = seed,
        Model = df$models[u] %>%
          str_remove_all("-0.25|-le_0.3-le_0.3|_0.3|RUE|ALSE") %>%
          str_remove("T-") %>%
          paste0(., "\nS=", seed) %>%
          as.factor(),
        Run = as.character(y) %>% as.factor(),
        Epoch = 1:length(history_trains[[df$models[u]]]$metrics$loss),
        steps = history_trains[[df$models[u]]]$params$steps
      )
    }) %>% do.call(rbind.data.frame, .)
  }) %>%
    do.call(rbind.data.frame, .) %>%
    as_tibble() %>%
    pivot_longer(c("Train", "Validation"), names_to = "Data", values_to = "loss")

  train_plot <- ggplot(df2, aes(y = loss, x = Epoch, color = Data)) +
    geom_line(size = 0.4) +
    facet_grid(vars(Run), vars(Model), scales = "free") +
    scale_y_continuous(
      limits = c(1.1, 1.7), labels = c("1.2", "1.4", "1.6"),
      breaks = c(1.2, 1.4, 1.6)
    ) +
    scale_x_continuous(breaks = breaks_pretty(n = 3)) +
    theme_light(base_size = 9, base_family = "Helvetica") +
    labs(y = "Bernoulli Gamma loss function", color = "") +
    scale_colour_manual(values = c("#0073c2", "#cd2926")) +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1),
      panel.spacing = unit(1.5, "mm"),
      strip.background = element_rect(fill = "white"),
      strip.text = element_text(color = "black"),
      strip.text.x = element_text(margin = margin(c(0, 0, 1, 0), unit = "mm")),
      strip.text.y = element_text(margin = margin(c(0, 0, 0, 1), unit = "mm")),
      legend.margin = margin(-3.5, 0, -1.5, 0, unit = "mm"),
      legend.key.size = unit(6, "mm"),
      legend.key.width = unit(6, "mm"),
      legend.position = "bottom",
      plot.margin = margin(0, 0, 0, 0.25, unit = "mm")
    )

  ggsave(paste0(path, "/loss_plots/Repr_", plot_name, ".pdf"),
    plot = train_plot, width = 175, height = 120, units = "mm"
  )

  return(df2)
}
