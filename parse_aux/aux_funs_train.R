# Written by Dánnell Quesada-Chacón

# Function that creates the strings of all the combinations possible among the given hyperparameters
models_strings_fun <- function(models = c("", "pp"), layers = c(3, 4, 5), F_first = c(16, 32, 64, 128), F_last = c(1, 3), do_u = c(TRUE), do_rate = c(0.25), dense_after_u = c(FALSE), dense_units = list(c(256, 128), c(128)), activ_main = c("lelu"), activ_last = c("lelu"), alpha_main = c(0.3), alpha_last = c(0.3), BN_main = c(TRUE), BN_last = c(FALSE, TRUE)) {
  unique(unlist(sapply(F_first, FUN = function(z) {
    sapply(F_last, FUN = function(y) {
      sapply(do_u, FUN = function(x) {
        sapply(do_rate, FUN = function(w) {
          sapply(dense_after_u, FUN = function(v) {
            sapply(dense_units, FUN = function(u) {
              sapply(models, FUN = function(t) {
                sapply(layers, FUN = function(o) {
                  sapply(activ_last, FUN = function(s) {
                    sapply(activ_main, FUN = function(r) {
                      sapply(BN_main, FUN = function(m) {
                        sapply(BN_last, FUN = function(n) {
                          sapply(alpha_main, FUN = function(q) {
                            sapply(alpha_last, FUN = function(p) {
                              if (r == "lelu" | r == "leaky_relu") {
                                activ_s1 <- paste0(substr(r, 1, 2), "_", q)
                                alpha_s1 <- paste0(", alpha1 = ", q)
                              } else {
                                activ_s1 <- substr(r, 1, 2)
                                alpha_s1 <- ""
                              }
                              if (s == "lelu" | s == "leaky_relu") {
                                activ_s2 <- paste0(substr(s, 1, 2), "_", p)
                                alpha_s2 <- paste0(", alpha2 = ", p)
                              } else {
                                activ_s2 <- substr(s, 1, 2)
                                alpha_s2 <- ""
                              }
                              if (x) {
                                if (v) {
                                  list(
                                    paste(paste0("U", t), o, z, y, w, activ_s1, activ_s2, paste(u, collapse = "_"), m, n, sep = "-"),
                                    paste0("model <- model_unet", t, "(", o, ", ", z, ", activation = '", r, "', input_shape = input_shape, output_shape = output_shape, filters_last = ", y, ", act_last = '", s, "', BaNorm1 = ", m, ", BaNorm2 = ", n, ", DropOut = ", x, ", spatialDOrate = ", w, ", dense = ", v, ", dense_units = ", list(u), alpha_s1, alpha_s2, ")")
                                  )
                                } else {
                                  list(
                                    paste(paste0("U", t), o, z, y, w, activ_s1, activ_s2, m, n, sep = "-"),
                                    paste0("model <- model_unet", t, "(", o, ", ", z, ", activation = '", r, "', input_shape = input_shape, output_shape = output_shape, filters_last = ", y, ", act_last = '", s, "', BaNorm1 = ", m, ", BaNorm2 = ", n, ", DropOut = ", x, ", spatialDOrate = ", w, ", dense = ", v, alpha_s1, alpha_s2, ")")
                                  )
                                }
                              } else if (v) {
                                list(
                                  paste(paste0("U", t), o, z, y, activ_s1, activ_s2, paste(u, collapse = "_"), m, n, sep = "-"),
                                  paste0("model <- model_unet", t, "(", o, ", ", z, ", activation = '", r, "', input_shape = input_shape, output_shape = output_shape, filters_last = ", y, ", act_last = '", s, "', BaNorm1 = ", m, ", BaNorm2 = ", n, ", DropOut = ", x, ", dense = ", v, ", dense_units = ", list(u), alpha_s1, alpha_s2, ")")
                                )
                              } else {
                                list(
                                  paste(paste0("U", t), o, z, y, activ_s1, activ_s2, m, n, sep = "-"),
                                  paste0("model <- model_unet", t, "(", o, ", ", z, ", activation = '", r, "', input_shape = input_shape, output_shape = output_shape, filters_last = ", y, ", act_last = '", s, "', BaNorm1 = ", m, ", BaNorm2 = ", n, ", DropOut = ", x, ", dense = ", v, alpha_s1, alpha_s2, ")")
                                )
                              }
                            }, simplify = T)
                          }, simplify = T)
                        }, simplify = T)
                      }, simplify = T)
                    }, simplify = T)
                  }, simplify = T)
                }, simplify = T)
              }, simplify = T)
            }, simplify = T)
          }, simplify = T)
        }, simplify = T)
      }, simplify = T)
    }, simplify = T)
  }, simplify = T), recursive = F))
}

# Function which receives the strings of the architecture and returns the models
architectures <- function(architecture, input_shape, output_shape) {
  if (architecture == "CNN1") {
    inputs <- layer_input(shape = input_shape)

    l <- conv2d_stack(inputs, filters = c(50, 25, 1), activation = "relu", kernel_size = 3) %>%
      layer_flatten()

    outputs <- param_bernoulli(l, output_shape = output_shape)
    model <- keras_model(inputs = inputs, outputs = outputs)
  }

  if (architecture == "CNN64-1") {
    inputs <- layer_input(shape = input_shape)
    x <- inputs

    l <- conv2d_stack(inputs, filters = c(64, 32, 16, 1), activation = "relu", kernel_size = 3) %>%
      layer_flatten()

    outputs <- param_bernoulli(l, output_shape = output_shape)
    model <- keras_model(inputs = inputs, outputs = outputs)
  }

  if (architecture == "CNN32-1") {
    inputs <- layer_input(shape = input_shape)
    x <- inputs

    l <- conv2d_stack(inputs, filters = c(32, 16, 1), activation = "relu", kernel_size = 3) %>%
      layer_flatten()

    outputs <- param_bernoulli(l, output_shape = output_shape)
    model <- keras_model(inputs = inputs, outputs = outputs)
  }

  if (architecture == "CNN64_3-1") {
    inputs <- layer_input(shape = input_shape)
    x <- inputs

    l <- conv2d_stack(inputs, filters = c(64, 32, 1), activation = "relu", kernel_size = 3) %>%
      layer_flatten()

    outputs <- param_bernoulli(l, output_shape = output_shape)
    model <- keras_model(inputs = inputs, outputs = outputs)
  }

  if (architecture %in% models_strings) {
    eval(parse(text = models_strings[(match(architecture, models_strings) + 1)]))
  }
  return(model)
}

# Small modification of downscaleTrain.keras to return training history
downscaleTrain.keras.mod <- function(obj, model, compile.args = list(object = model), fit.args = list(object = model), clear.session = FALSE) {
  compile.args[["object"]] <- model
  if (is.null(compile.args[["optimizer"]])) {
    compile.args[["optimizer"]] <- optimizer_adam()
  }
  if (is.null(compile.args[["loss"]])) {
    compile.args[["loss"]] <- "mse"
  }
  do.call(compile, args = compile.args)
  fit.args[["object"]] <- model
  fit.args[["x"]] <- obj$x.global
  fit.args[["y"]] <- obj$y$Data
  history <- do.call(fit, args = fit.args)
  if (isTRUE(clear.session)) {
    k_clear_session()
  } else {
    model
  }
  return(history)
}

# Function that comprises the functions needed to predict precipitation
predict_out <- function(xy.t, xy.tT, model, yT_bin, loss = "bernouilliGammaLoss", C4R.template) {
  out <- downscalePredict.keras(xy.t, model, C4R.template = C4R.template, loss = loss)
  aux2 <- downscalePredict.keras(xy.tT, model, C4R.template = C4R.template, loss = loss) %>% subsetGrid(var = "p")
  aux1 <- subsetGrid(out, var = "p")
  bin <- binaryGrid(aux1, ref.obs = yT_bin, ref.pred = aux2)
  bin$Variable$varName <- "bin"
  out <- makeMultiGrid(
    subsetGrid(out, var = "p"),
    subsetGrid(out, var = "log_alpha"),
    subsetGrid(out, var = "log_beta"),
    bin
  ) %>% redim(member = FALSE)
  return(out)
}
