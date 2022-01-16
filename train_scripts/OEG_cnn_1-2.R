# Written by Dánnell Quesada-Chacón
# Based on Baño-Medina et al., 2020 (https://doi.org/10.5194/gmd-13-2109-2020)

seed <- Sys.getenv("PYTHONHASHSEED")
n_repeat <- as.integer(commandArgs(trailingOnly = TRUE)[1])

library(reticulate)
library(downscaleR.keras)

np <- import("numpy")

np <- import("numpy") # Sometimes needs to be done twice for it to load properly
rd <- import("random")

# Function to reset all seeds
reset_seeds <- function(seed = 42) {
  np$random$seed(seed)
  rd$seed(seed)
  tf$random$set_seed(seed)
  set.seed(seed)
}

reset_seeds(as.integer(seed))

library(loadeR)
library(transformeR)
library(climate4R.value)
library(magrittr)
library(gridExtra)
library(sp)

tensorflow::tf$config$experimental$list_physical_devices("GPU")
physical_devices <- tensorflow::tf$config$experimental$list_physical_devices("GPU")

# Allow memory growth for GPUs
sapply(1:length(physical_devices), function(x) {
  tf$config$experimental$set_memory_growth(physical_devices[[x]], TRUE)
  print(physical_devices[[x]])
})

try(k_constant(1), silent = TRUE) # Small test for the GPU
try(k_constant(1), silent = TRUE) # Sometimes needs to be done twice for it to load properly

path <- "./"

setwd(path)

source("./unet_def.R") # File with functions for U architectures
source("./aux_funs_train.R") # File with functions to ease traning process
dir.create(paste0("Data/precip/", seed), recursive = TRUE)
dir.create(paste0("models/precip/", seed), recursive = TRUE)

# To see full output of summary()
options("width" = 150)
options(warn = -1)

load("./Data/precip/x_32.rda")

xT <- subsetGrid(x, years = 1979:2009)
xt <- subsetGrid(x, years = 2010:2015)
xt <- scaleGrid(xt, xT, type = "standardize", spatial.frame = "gridbox") %>% redim(drop = TRUE)
xT <- scaleGrid(xT, type = "standardize", spatial.frame = "gridbox") %>% redim(drop = TRUE)
gc()

load(file = paste0("./Data/precip", "/y_OEG.rda"))

yT <- subsetGrid(y, years = 1979:2009)
yT_bin <- binaryGrid(yT, threshold = 1, condition = "GE")
yt <- subsetGrid(y, years = 2010:2015)
yt_bin <- binaryGrid(yt, threshold = 1, condition = "GE")
gc()

xy.T <- prepareData.keras(xT, binaryGrid(gridArithmetics(yT, 1, operator = "-"),
  condition = "GE",
  threshold = 0,
  partial = TRUE
),
first.connection = "conv",
last.connection = "dense",
channels = "last"
)
xy.tT <- prepareNewData.keras(xT, xy.T)
xy.t <- prepareNewData.keras(xt, xy.T)
gc()

# Benchmark model CNN1 and variations of it
deepName <- c("CNN1")

u_model <- c("", "pp") # Type of u model, "" for U-Net, "pp" for U-Net++
u_layers <- c(3, 4, 5) # Depth of the u model
u_seeds <- c(16, 32, 64, 128) # Number of filters of the first layer
u_Flast <- c(1, 3) # Number of filters of the last ConvUnit
u_do <- c(TRUE) # Boolean for dropout within the u model
u_spdor <- c(0.25) # Fraction of the spatial dropout within the u model
u_dense <- c(FALSE) # Boolean for dense units after ConvUnit
u_dunits <- list(c(256, 128), c(128)) # Dense units, passed as a list, several layers supported
act_main <- c("lelu") # Activation function within u model and dense units
act_last <- c("lelu") # Activation function for the last ConvUnit
alpha1 <- c(0.3) # If leaky relu (lelu) used, which alpha. For u model and dense units
alpha2 <- c(0.3) # Same as above but for last ConvUnit
BN1 <- c(TRUE) # Batch normalization inside u model
BN2 <- c(FALSE, TRUE) # Batch normalization for the last ConvUnit

models_strings <- models_strings_fun(
  models = u_model, layers = u_layers, F_first = u_seeds,
  F_last = u_Flast, do_u = u_do, do_rate = u_spdor,
  dense_after_u = u_dense, dense_units = u_dunits,
  activ_main = act_main, activ_last = act_last,
  alpha_main = alpha1, alpha_last = alpha2,
  BN_main = BN1, BN_last = BN2
)

# deepName <- c(deepName, models_strings[seq(1, length(models_strings), 2)])

# Type of simulations
simulateName <- c("deterministic", "stochastic")
simulateDeep <- c(FALSE, TRUE)

# The next 2 parameters should be changed according to the GPU memory
batch_size <- 512
lr <- 0.0005
patience <- 75
validation_split <- 0.1

times_repeat <- seq_len(n_repeat) # To repeat n times
to_train <- (1:length(deepName))

sapply(times_repeat, simplify = FALSE, FUN = function(n) {
  history_trains <- list()

  print(paste0("Repetition #", n, " of ", n_repeat))

  sapply(to_train, simplify = FALSE, FUN = function(z) {
    print(paste0("Model ", z, " out of ", length(to_train), " = ", deepName[z]))
    gc()

    # For multi GPU, repeatability not fully tested
    if (length(physical_devices) > 1) {
      reset_seeds(as.integer(seed))
      strategy <- tf$distribute$MirroredStrategy()
      print(paste0("Number of GPUs on which the model will be distributed: ", strategy$num_replicas_in_sync))
      reset_seeds(as.integer(seed))
      with(strategy$scope(), {
        model <- architectures(
          architecture = deepName[z],
          input_shape = dim(xy.T$x.global)[-1],
          output_shape = dim(xy.T$y$Data)[2]
        )
      })
      summary(model)

      reset_seeds(as.integer(seed))
      history_train <- downscaleTrain.keras.mod(
        obj = xy.T,
        model = model,
        clear.session = TRUE,
        compile.args = list(
          "loss" = bernouilliGammaLoss(last.connection = "dense"),
          "optimizer" = optimizer_adam(lr = lr)
        ),
        fit.args = list(
          "batch_size" = batch_size,
          "epochs" = 5000,
          "validation_split" = validation_split,
          "verbose" = 1,
          "callbacks" = list(
            callback_early_stopping(patience = patience),
            callback_model_checkpoint(
              filepath = paste0("./models/precip/", seed, "/", deepName[z], "_", n, ".h5"),
              monitor = "val_loss", save_best_only = TRUE
            )
          )
        )
      )
    } else {
      # For one GPU
      reset_seeds(as.integer(seed))
      model <- architectures(
        architecture = deepName[z],
        input_shape = dim(xy.T$x.global)[-1],
        output_shape = dim(xy.T$y$Data)[2]
      )
      summary(model)

      reset_seeds(as.integer(seed))
      history_train <- downscaleTrain.keras.mod(
        obj = xy.T,
        model = model,
        clear.session = TRUE,
        compile.args = list(
          "loss" = bernouilliGammaLoss(last.connection = "dense"),
          "optimizer" = optimizer_adam(lr = lr)
        ),
        fit.args = list(
          "batch_size" = batch_size,
          "epochs" = 5000,
          "validation_split" = validation_split,
          "verbose" = 1,
          "callbacks" = list(
            callback_early_stopping(patience = patience),
            callback_model_checkpoint(
              filepath = paste0("./models/precip/", seed, "/", deepName[z], "_", n, ".h5"),
              monitor = "val_loss", save_best_only = TRUE
            )
          )
        )
      )
    }
    gc()

    hist_t <- list(history_train)
    names(hist_t) <- deepName[z]

    history_trains <<- append(history_trains, hist_t)

    # Save history after training each model
    save(history_trains, file = paste0("./Data/precip/", seed, "/hist_train_CNN_", n, ".rda"))

    pred <- predict_out(xy.t, xy.tT,
      model = model,
      yT_bin, loss = "bernouilliGammaLoss", C4R.template = yT
    )

    # Predict rainfall
    sapply(1:length(simulateDeep), simplify = FALSE, FUN = function(zz) {
      print(simulateName[zz])

      reset_seeds(as.integer(seed))
      pred_amo <- computeRainfall(
        log_alpha = subsetGrid(pred, var = "log_alpha"),
        log_beta = subsetGrid(pred, var = "log_beta"),
        bias = 1,
        simulate = simulateDeep[zz]
      )
      pred_ocu <- subsetGrid(pred, var = "p") %>% redim(drop = TRUE)
      pred_bin <- subsetGrid(pred, var = "bin")
      pred_serie <- gridArithmetics(pred_amo, pred_bin)

      save(pred_bin, pred_ocu, pred_amo, pred_serie,
        file = paste0("./Data/precip/", seed, "/predictions_", simulateName[zz], "_", deepName[z], "_", n, ".rda")
      )
      gc()
    })
  })
})

gc()

# Calculate metrics parallelly
library(doParallel)
# cores <- getOption("cl.cores", detectCores()) # to use all cores, memory issues may arise
cores <- 6 # Parallel computing for six cores, change accordingly

simulateName <- c(rep("deterministic", 5), "stochastic", rep("deterministic", 3))
models <- c(deepName)
measures <- c("ts.rocss", "ts.RMSE", "ts.rs", rep("biasRel", 3), rep("bias", 3))
index <- c(rep(NA, 3), "Mean", rep("P98", 2), "AnnualCycleRelAmp", "WetAnnualMaxSpell", "DryAnnualMaxSpell")
cl <- makeCluster(mc <- cores)
clusterExport(
  cl = cl, varlist = c("simulateName", "models", "measures", "index", "seed", "yt_bin", "yt", 
    "times_repeat", "deepName", "n_repeat", "cores"),
  envir = environment()
)
cl.libs <- clusterEvalQ(cl = cl, expr = {
  library(reticulate)
  library(downscaleR.keras)
  library(loadeR)
  library(transformeR)
  library(climate4R.value)
  library(magrittr)
  library(gridExtra)
  library(sp)
  library(doParallel)
})

# Trying to optimize the cores to Calculate the metrics in Parallel
if (n_repeat < cores) {
  if (length(models) < cores) { # & cores < length(measures)
    sapply(times_repeat, FUN = function(n) {
      validation.list <- parSapply(cl = cl, 1:length(measures), simplify = FALSE, FUN = function(z) {
        sapply(1:length(models), FUN = function(zz) {
          args <- list()
          load(paste0("./Data/precip/", seed, "/predictions_", simulateName[z], "_", models[zz], "_", n, ".rda"))
          if (simulateName[z] == "deterministic") {
            pred_serie <- gridArithmetics(pred_amo, pred_bin)
            if (measures[z] == "ts.rocss") {
              args[["y"]] <- yt_bin
              args[["x"]] <- pred_ocu
            } else if (measures[z] == "ts.RMSE") {
              args[["y"]] <- yt
              args[["x"]] <- pred_amo
              args[["condition"]] <- "GE"
              args[["threshold"]] <- 1
              args[["which.wetdays"]] <- "Observation"
            } else {
              args[["y"]] <- yt
              args[["x"]] <- pred_serie
            }
          } else {
            pred_serie <- gridArithmetics(pred_amo, pred_bin)
            args[["y"]] <- yt
            args[["x"]] <- pred_serie
          }
          args[["measure.code"]] <- measures[z]
          if (!is.na(index[z])) args[["index.code"]] <- index[z]
          do.call("valueMeasure", args)$Measure
        }, simplify = F) %>% makeMultiGrid()
      })

      validation.list <- append(validation.list, list(deepName))
      save(validation.list, file = paste0("./Data/precip/", seed, "/validation_CNN_", n, ".rda"))
    })

    stopCluster(cl = cl)
  } else {
    sapply(times_repeat, FUN = function(n) {
      validation.list <- sapply(1:length(measures), simplify = FALSE, FUN = function(z) {
        parSapply(cl = cl, 1:length(models), FUN = function(zz) {
#           sapply(1:length(models), FUN = function(zz) {
            args <- list()
            load(paste0("./Data/precip/", seed, "/predictions_", simulateName[z], "_", models[zz], "_", n, ".rda"))
            if (simulateName[z] == "deterministic") {
              pred_serie <- gridArithmetics(pred_amo, pred_bin)
              if (measures[z] == "ts.rocss") {
                args[["y"]] <- yt_bin
                args[["x"]] <- pred_ocu
              } else if (measures[z] == "ts.RMSE") {
                args[["y"]] <- yt
                args[["x"]] <- pred_amo
                args[["condition"]] <- "GE"
                args[["threshold"]] <- 1
                args[["which.wetdays"]] <- "Observation"
              } else {
                args[["y"]] <- yt
                args[["x"]] <- pred_serie
              }
            } else {
              pred_serie <- gridArithmetics(pred_amo, pred_bin)
              args[["y"]] <- yt
              args[["x"]] <- pred_serie
            }
            args[["measure.code"]] <- measures[z]
            if (!is.na(index[z])) args[["index.code"]] <- index[z]
            do.call("valueMeasure", args)$Measure
          }, simplify = F) %>% makeMultiGrid()
        })

        validation.list <- append(validation.list, list(deepName))
        save(validation.list, file = paste0("./Data/precip/", seed, "/validation_CNN_", n, ".rda"))
      })
    }

  stopCluster(cl = cl)
} else {
  parSapply(cl = cl, times_repeat, FUN = function(n) {
    validation.list <- sapply(1:length(measures), simplify = FALSE, FUN = function(z) {
      sapply(1:length(models), FUN = function(zz) {
        args <- list()
        load(paste0("./Data/precip/", seed, "/predictions_", simulateName[z], "_", models[zz], "_", n, ".rda"))
        if (simulateName[z] == "deterministic") {
          pred_serie <- gridArithmetics(pred_amo, pred_bin)
          if (measures[z] == "ts.rocss") {
            args[["y"]] <- yt_bin
            args[["x"]] <- pred_ocu
          } else if (measures[z] == "ts.RMSE") {
            args[["y"]] <- yt
            args[["x"]] <- pred_amo
            args[["condition"]] <- "GE"
            args[["threshold"]] <- 1
            args[["which.wetdays"]] <- "Observation"
          } else {
            args[["y"]] <- yt
            args[["x"]] <- pred_serie
          }
        } else {
          pred_serie <- gridArithmetics(pred_amo, pred_bin)
          args[["y"]] <- yt
          args[["x"]] <- pred_serie
        }
        args[["measure.code"]] <- measures[z]
        if (!is.na(index[z])) args[["index.code"]] <- index[z]
        do.call("valueMeasure", args)$Measure
      }, simplify = F) %>% makeMultiGrid()
    })

    validation.list <- append(validation.list, list(deepName))
    save(validation.list, file = paste0("./Data/precip/", seed, "/validation_CNN_", n, ".rda"))
  })

  stopCluster(cl = cl)
}
