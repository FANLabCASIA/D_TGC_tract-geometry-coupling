rm(list = ls())# 加载必要的库avg_count_800ori1
library(magrittr)
library(dplyr)
library(mgcv)
library(ggplot2)

# 读取 df_fdc 数据final_paper_1st_affinity
# df_fdc <- read.csv("E:\\hcpd\\final_paper_1st_10dis_FD.csv")
df_fdc <- read.csv("/Users/deyingli/Documents/code/R_dir/Trajectory/cleaned_data_1_2_L1.csv")
# 获取所有 Mode1 到 Mode360 的列名
expected_columns <- paste0("Mode", 1:36)

# 过滤 df_fdc 中实际存在的列
existing_columns <- intersect(expected_columns, names(df_fdc))

# 重新排列 df_fdc 中的列，按照 Mode1 到 Mode360 的顺序
df_fdc <- df_fdc %>%
  dplyr::select(all_of(existing_columns), everything())

# 获取组件编号
comp_names <- df_fdc %>%
  dplyr::select(Mode1:Mode36)
Components <- names(comp_names)

# 转换组件为整数
createInteger <- function(f) {
  as.numeric(as.character(f))
}
comp_names <- as.data.frame(mapply(createInteger, comp_names))

# 列出组件名称
bundles <- paste0("Mode", 1:36)
bundles <- as.data.frame(bundles)

########################################
#### BUILD GAM MODELS - AGE EFFECTS ####
########################################

# 构建全模型
# 创建 GAM 模型列表
gamModels_age_fdc <- lapply(Components, function(x) { 
  gam(substitute(i ~ s(interview_age, k=3) + sex, list(i = as.name(x))), method = "REML", data = df_fdc)
})

# 查看模型摘要
models_age_fdc <- lapply(gamModels_age_fdc, summary)

# 提取 F 统计量
fstat_age_fdc <- sapply(gamModels_age_fdc, function(v) summary(v)$s.table[3])
fstat_age_fdc <- as.data.frame(fstat_age_fdc)

# 提取年龄 p 值
p_age_fdc <- sapply(gamModels_age_fdc, function(v) summary(v)$s.table[4])
p_age_fdc <- as.data.frame(p_age_fdc)
p_age_fdc <- round(p_age_fdc, 3)

# FDR 校正 p 值
p_age_fdc_fdr <- as.data.frame(p.adjust(p_age_fdc[,1], method = "bonferroni"))
p_age_fdc_fdr <- round(p_age_fdc_fdr, 3)

# 提取 R 方值
r_squared <- sapply(gamModels_age_fdc, function(v) summary(v)$r.sq)
r_squared <- as.data.frame(r_squared)
r_squared <- round(r_squared, 3)

# 结合 F 统计量和 FDR 校正 p 值
age_gam <- cbind(bundles, fstat_age_fdc, p_age_fdc_fdr,r_squared)
colnames(age_gam) <- c("bundles", "fstat_age_fdc", "p_age_fdc_fdr","r_squared")

# 获取部分 R2（作为效应量的度量）基于 Chenying 的脚本
modes <- paste0("Mode", 1:36)
df_fdc_min <- df_fdc %>%
  dplyr::select(modes, interview_age, sex)

# 构建简化模型
redmodel_fdc <- lapply(Components, function(x) {
  gam(substitute(i ~ sex , list(i = as.name(x))), method = "REML", data = df_fdc_min)
})

# 计算部分 R2
partialRsq <- function(fullmodel, redmodel) {
  sse.full <- sum((fullmodel$y - fullmodel$fitted.values)^2)
  sse.red <- sum((redmodel$y - redmodel$fitted.values)^2)
  
  partialRsq <- (sse.red - sse.full) / sse.red
  
  toReturn <- list(partialRsq = partialRsq, sse.full = sse.full, sse.red = sse.red)
  return(toReturn)
}

# 合并部分 R2 值与 F 统计量和 p 值
partialR2_list <- lapply(1:36, function(i) {
  partialRsq(gamModels_age_fdc[[i]], redmodel_fdc[[i]])$partialRsq
})

partialR2 <- do.call(cbind, partialR2_list)
colnames(partialR2) <- paste0("Mode", 1:36)
partialR2 <- as.data.frame(t(partialR2))
partialR2 <- partialR2 %>%
  dplyr::rename(partial_R2 = V1)

# 合并并排序
age_gam <- cbind(age_gam, partialR2)
age_gam$partial_R2 <- as.numeric(as.character(age_gam$partial_R2))
age_gam <- age_gam[order(-age_gam$partial_R2),]

# 保存最终数据
# write.csv(age_gam, file = "E:\\hcpd\\r_gamrelated\\final_paper_1st_10dis_gam.csv", row.names = FALSE)
write.csv(age_gam, file = "./Trajectory/smoothed_curves_AveAlongMode_L1/final_paper_1st_10dis_gam.csv", row.names = FALSE)

# 函数: 计算每个模式的 GAM 模型平均值，不进行平滑
compute_mode_average <- function(models, mode_names) {
  combined_data <- do.call(rbind, lapply(1:length(models), function(i) {
    model <- models[[i]]
    mode_name <- mode_names[i]
    
    complete_data <- model.frame(model)
    data.frame(Age = complete_data$interview_age,
               ModeValue = complete_data[[mode_name]],
               ModeName = mode_name)
  }))
  
  combined_data <- combined_data %>%
    group_by(Age, ModeName) %>%
    summarise(AvgModeValue = mean(ModeValue, na.rm = TRUE)) %>%
    ungroup()
  
  combined_data
}

# 获取每个模式的平均值，不进行平滑
mode_avg_data <- compute_mode_average(gamModels_age_fdc, Components)

# 保存平均值到 CSV 文件
# write.csv(mode_avg_data, file = "E:\\hcpd\\r_gamrelated\\final_paper_1st_10dis_mean.csv", row.names = FALSE)
write.csv(mode_avg_data, file = "./Trajectory/smoothed_curves_AveAlongMode_L1/final_paper_1st_10dis_mean.csv", row.names = FALSE)

# 使用 mgcv 包和 gam 模型来直接计算一阶和二阶导数
library(mgcv)

# 使用 mgcv 包和 gam 模型来直接计算一阶和二阶导数
library(gratia)
save_smoothed_curves <- function(models, mode_names, image_output_dir, csv_output_dir, smoothed_output_file, df_fdc, draws = 1000, increments = 168, smooth_var = "interview_age", knots = 3) {  
  dir.create(image_output_dir, showWarnings = FALSE)  # 创建图像输出目录
  dir.create(csv_output_dir, showWarnings = FALSE)    # 创建 CSV 输出目录

  all_smoothed_data <- do.call(rbind, lapply(seq_along(mode_names), function(i) {
    mode_name <- mode_names[i]
    
    # 生成新的模型 (使用 gam 建立模型)
    formula <- as.formula(paste(mode_name, "~ s(interview_age, k=3) + sex"))
    gam_model <- gam(formula, method = "REML", data = df_fdc)

    # 提取数据（这里假设原数据集 df_fdc 中包含相关字段）
    data <- data.frame(interview_age = df_fdc$interview_age,
                       ModeValue = df_fdc[[mode_name]],
                       sex = df_fdc$sex)

    # 输出 geom_point 所绘制的点的数量
    cat("Number of points for", mode_name, ":", nrow(data), "\n")

    # 获取 sex 和 motion 的常见值或中位数
    most_common_sex <- names(sort(table(data$sex), decreasing = TRUE))[1]  # sex 的最常见水平
    # median_motion <- median(data$motion, na.rm = TRUE)  # motion 的中位数

    # 创建预测数据框，这里固定 sex 和 motion 的值
    new_data <- data.frame(
      interview_age = seq(min(data$interview_age), max(data$interview_age), length.out = increments),
      sex = most_common_sex  # 将 sex 固定为常见值
      # motion = median_motion   # 将 motion 固定为中位数
    )

    # 使用 gam 模型进行预测，计算平滑值和标准误
    pred <- fitted_values(object =gam_model, data = new_data)
    # print(pred)
    new_data$SmoothedValue <- pred$.fitted
    new_data$UpperCI <- pred$.upper_ci  # 95%置信区间
    new_data$LowerCI <- pred$.lower_ci
    new_data$ModeName <- mode_name

    # 使用 derivatives() 计算平滑项的一阶导数（Derivative）
    # Step 1: 计算平滑项的一阶导数（Derivative）
    derv1 <- derivatives(gam_model, newdata = new_data, term = sprintf("s(%s)", smooth_var))
    new_data$Derivative <- derv1$.derivative
    # print(names(derv1))
    # Step 2: 根据置信区间判断显著性
    # 判断导数显著性：如果置信区间不包含零，则认为导数显著
    derv1 <- derv1 %>% #add "sig" column (TRUE/FALSE) to derv
      mutate(sig = !(0 >  derv1$.lower_ci& 0 <  derv1$.upper_ci)) #derivative is sig if the lower CI is not < 0 while the upper CI is > 0 (i.e., when the CI does not include 0)
    derv1$sig_deriv = derv1$.derivative*derv1$sig #add "sig_deriv derivatives column where non-significant derivatives are set to 0

    # Step 4: 查找首次包含0的年龄点作为成熟年龄
    maturation_age <- NA
    if (sum(derv1$sig) > 0) {
      # 如果存在显著导数，找到最后一个显著导数对应的年龄
      maturation_age <- max(new_data$interview_age[derv1$sig == TRUE])  # 找到最后一个显著导数的年龄
    } else {
      # 如果没有显著导数，返回 NA
      maturation_age <- NA
    }


    # 如果未找到，则 MaturationAge 保持为 NA
    if (is.na(maturation_age)) {
      # 如果没有找到置信区间跨越零的点，则通过导数为零的点计算MaturationAge
      if (sum(new_data$Derivative == 0) > 0) {
        maturation_age <- max(new_data$interview_age[new_data$Derivative == 0])
      }
    }

    # 使用 derivatives() 计算平滑项的二阶导数（Curvature）
    derv2 <- derivatives(gam_model, newdata = new_data, term = sprintf("s(%s)", smooth_var), order = 2)
    new_data$Curvature <- derv2$.derivative

    # 确保一阶导数和二阶导数不同
    if (all(new_data$Derivative == new_data$Curvature)) {
      warning(paste("One and two derivatives are identical for", mode_name))
    }

    # 计算发育特征：Change Onset，Peak Change，下降结束的年龄，Maturation Age
    change_onset <- NA
    peak_change <- NA
    decrease_onset <- NA
    increase_onset <- NA  # 添加缺失的变量
    # 找到第一个显著导数变化的年龄点（Change Onset）
    if (sum(new_data$Derivative != 0) > 0) {
      change_onset <- min(new_data$interview_age[new_data$Derivative != 0])
    }

    # 找到最大导数值（Peak Change）
    if (sum(new_data$Derivative != 0) > 0) {
      peak_change <- new_data$interview_age[which.max(abs(new_data$Derivative))]
    }

    # 找到首次出现下降的年龄点（下降结束的年龄）
    if (sum(new_data$Derivative < 0) > 0) {
      decrease_onset <- min(new_data$interview_age[new_data$Derivative < 0])
    }

    # 找到首次出现上升的年龄点（上升开始的年龄）
    if (sum(new_data$Derivative > 0) > 0) {
      increase_onset <- min(new_data$interview_age[new_data$Derivative > 0])
    } else {
      increase_onset <- NA  # 如果没有上升的导数，设置为 NA
    }

    # 保存平滑曲线、一阶导数、二阶导数数据到 CSV 文件
    write.csv(new_data, file = file.path(csv_output_dir, paste0(mode_name, "_smooth_and_derivative_and_curvature_data.csv")), row.names = FALSE)

    # 根据模式选择颜色
    # line_color <- if (mode_name == "Mode267") {
    #   "#52C4F3"
    # } else if (mode_name == "Mode8") {
    #   "#A657E2"
    # } else {
    #   "blue"
    # }

    # # 绘制平滑曲线并保存图像
    # p <- ggplot() +
    #   geom_point(data = data, aes(x = interview_age, y = ModeValue), alpha = 0.3) +
    #   geom_line(data = new_data, aes(x = interview_age, y = SmoothedValue), color = line_color) +
    #   geom_ribbon(data = new_data, aes(x = interview_age, ymin = LowerCI, ymax = UpperCI), fill = line_color, alpha = 0.2) +
    #   labs(title = paste("Smooth Curve for", mode_name),
    #        x = "Age",
    #        y = "Drops") +  # 将纵坐标名称统一为 Drops
    #   theme_minimal() +
    #   theme(
    #     text = element_text(family = "Arial", size = 16),
    #     panel.grid = element_blank(),
    #     axis.line = element_line(color = "black"),
    #     plot.background = element_rect(fill = "white", color = NA),  # 设置背景为白色
    #     panel.background = element_rect(fill = "white", color = NA)  # 设置面板背景为白色
    #   )

    # # 保存图像到图像输出文件夹
    # ggsave(filename = file.path(image_output_dir, paste0(mode_name, "_smooth_curve.png")), plot = p, width = 8, height = 6)

    # # 保存散点数据到CSV文件夹
    # write.csv(data, file = file.path(csv_output_dir, paste0(mode_name, "_scatter_data.csv")), row.names = FALSE)
    
    # # 保存平滑曲线数据到CSV文件夹
    # write.csv(new_data, file = file.path(csv_output_dir, paste0(mode_name, "_smooth_data.csv")), row.names = FALSE)
    
    # 保存发育特征数据（Change Onset，Peak Change，下降结束的年龄，Maturation Age）
    dev_features <- data.frame(
      ModeName = mode_name,
      ChangeOnset = change_onset,
      PeakChange = peak_change,
      DecreaseOnset = decrease_onset,
      IncreaseOnset = increase_onset,
      MaturationAge = maturation_age
    )
    write.csv(dev_features, file = file.path(csv_output_dir, paste0(mode_name, "_developmental_features.csv")), row.names = FALSE)

    # 如果模式是 mode8 或 mode267，添加一个额外的图像保存
    # if (mode_name %in% c("Mode8", "Mode267")) {
    #   ggsave(filename = file.path(image_output_dir, paste0(mode_name, "_extra_smooth_curve.png")), plot = p, width = 8, height = 6)
    # }

    new_data
  }))

  # 保存所有模式的平滑数据到单个 CSV 文件
  write.csv(all_smoothed_data, file = smoothed_output_file, row.names = FALSE)
}



# 设置图像和 CSV 文件的输出目录
# image_output_directory <- "E:\\hcpd\\r_gamrelated\\final_paper_1st_10dis_images"
# csv_output_directory <- "E:\\hcpd\\r_gamrelated\\final_paper_1st_10dis_smoothed_800csv"
# smoothed_output_file <- "E:\\hcpd\\r_gamrelated\\final_paper_1st_10dis_smoothed_800.csv"
image_output_directory <- "./Trajectory/smoothed_curves_AveAlongMode_L1/final_paper_1st_10dis_images"
csv_output_directory <- "./Trajectory/smoothed_curves_AveAlongMode_L1/final_paper_1st_10dis_smoothed_800csv"
smoothed_output_file <- "./Trajectory/smoothed_curves_AveAlongMode_L1/final_paper_1st_10dis_smoothed_800.csv"

# 调用函数保存平滑曲线图像和数据
save_smoothed_curves(gamModels_age_fdc, Components, image_output_directory, csv_output_directory, smoothed_output_file, df_fdc,draws = 1000,  # 生成后验分布的模拟次数
                     increments = 168,  # 用于生成平滑曲线的增量点数
                     smooth_var = "interview_age",  # 平滑变量为 interview_age
                     knots = 3)
library(tidyr)
save_smoothed_curves_der_posterior <- function(models, mode_names, image_output_dir, csv_output_dir, smoothed_output_file, df_fdc, draws = 1000, increments = 168, smooth_var = "interview_age", knots = 3) {   
  dir.create(image_output_dir, showWarnings = FALSE)  # 创建图像输出目录
  dir.create(csv_output_dir, showWarnings = FALSE)    # 创建 CSV 输出目录

  all_smoothed_data <- do.call(rbind, lapply(seq_along(mode_names), function(i) {
    mode_name <- mode_names[i]
    
    # 生成新的模型 (使用 gam 建立模型)
    formula <- as.formula(paste(mode_name, "~ s(interview_age, k=3) + sex"))
    gam_model <- gam(formula, method = "REML", data = df_fdc)

    # 提取数据（这里假设原数据集 df_fdc 中包含相关字段）
    data <- data.frame(interview_age = df_fdc$interview_age,
                       ModeValue = df_fdc[[mode_name]],
                       sex = df_fdc$sex,
                       motion = df_fdc$motion)

    # 输出 geom_point 所绘制的点的数量
    cat("Number of points for", mode_name, ":", nrow(data), "\n")

    # 设置参数
    npd <- as.numeric(draws)  # number of draws from the posterior distribution; number of posterior derivative sets estimated
    np <- as.numeric(increments)  # number of smooth_var increments to get derivatives at
    EPS <- 1e-07  # finite differences
    UNCONDITIONAL <- FALSE  # should we account for uncertainty when estimating smoothness parameters?

    # 提取模型数据
    df <- gam_model$model  # extract the data used to build the gam, i.e., a df of y + predictor values 

    # 创建预测数据框，用于估计（后验）模型系数
    thisPred <- data.frame(init = rep(0, np)) 

    theseVars <- attr(gam_model$terms, "term.labels")  # gam model predictors (smooth_var + covariates)
    varClasses <- attr(gam_model$terms, "dataClasses")  # classes of the model predictors and y measure
    thisResp <- as.character(gam_model$terms[[2]])  # the measure to fit for each posterior draw

    for (v in c(1:length(theseVars))) {  # fill the prediction df with data 
      thisVar <- theseVars[[v]]
      thisClass <- varClasses[thisVar]
      if (thisVar == smooth_var) { 
        thisPred[, smooth_var] = seq(min(df[, smooth_var], na.rm = TRUE), max(df[, smooth_var], na.rm = TRUE), length.out = np)  # generate a range of np data points
      } else if (thisVar == "sex" || thisVar == "motion") {
        # 使用中位数填充 sex 和 motion
        thisPred[, thisVar] = median(df[, thisVar], na.rm = TRUE)  # 填充 sex 和 motion
      } else {
        switch(thisClass,
               "numeric" = {thisPred[, thisVar] = median(df[, thisVar])},  # make predictions based on median value
               "factor" = {thisPred[, thisVar] = levels(df[, thisVar])[[1]]},  # make predictions based on first level of factor
               "ordered" = {thisPred[, thisVar] = levels(df[, thisVar])[[1]]}  # make predictions based on first level of ordinal variable
        )
      }
    }

    pred <- thisPred %>% select(-init)  # prediction df
    pred2 <- pred  # second prediction df
    pred2[, smooth_var] <- pred[, smooth_var] + EPS  # finite differences

    # 估计平滑项的导数
    derivs <- derivatives(gam_model, term = sprintf('s(%s)', smooth_var), interval = "simultaneous", unconditional = UNCONDITIONAL, newdata = pred)  # derivative at 200 indices of smooth_var with a simultaneous CI
    # print(derivs )
    derivs.fulldf <- derivs %>% select( .derivative, .se, .lower_ci, .upper_ci)
    derivs.fulldf <- derivs.fulldf %>% #add "sig" column (TRUE/FALSE) to derv
      mutate(sig = !(0 >  derivs.fulldf$.lower_ci& 0 <  derivs.fulldf$.upper_ci)) #derivative is sig if the lower CI is not < 0 while the upper CI is > 0 (i.e., when the CI does not include 0)
    derivs.fulldf$sig_deriv = derivs.fulldf$.derivative*derivs.fulldf$sig #add "sig_deriv derivatives column where non-significant derivatives are set to 0

    colnames(derivs.fulldf) <- c(sprintf("%s", smooth_var), "derivative", "se", "lower", "upper", "significant", "significant.derivative")

    # 估计后验平滑导数
    Vb <- vcov(gam_model, unconditional = UNCONDITIONAL)  # variance-covariance matrix for all the fitted model parameters
    sims <- MASS::mvrnorm(npd, mu = coef(gam_model), Sigma = Vb)  # simulate model parameters (coefficients) from the posterior distribution of the smooth
    X0 <- predict(gam_model, newdata = pred, type = "lpmatrix")  # get matrix of linear predictors for pred
    X1 <- predict(gam_model, newdata = pred2, type = "lpmatrix")  # get matrix of linear predictors for pred2
    Xp <- (X1 - X0) / EPS 
    posterior.derivs <- Xp %*% t(sims)  # Xp * simulated model coefficients = simulated derivatives. Each column of posterior.derivs contains derivatives for a different draw from the simulated posterior distribution
    posterior.derivs <- as.data.frame(posterior.derivs)
    colnames(posterior.derivs) <- sprintf("draw%s", seq(from = 1, to = npd))  # label the draws
    posterior.derivs <- cbind(as.numeric(pred[, smooth_var]), posterior.derivs)  # add smooth_var increments from pred df to first column
    colnames(posterior.derivs)[1] <- sprintf("%s", smooth_var)  # label the smooth_var column
    posterior.derivs <- cbind(as.character(mode_name), posterior.derivs)  # add parcel label to first column
    colnames(posterior.derivs)[1] <- "label"  # label the column
    posterior.derivs.long <- posterior.derivs %>% pivot_longer(contains("draw"), names_to = "draw", values_to = "posterior.derivative")

    # 输出后验导数的数据
    cat("Posterior derivatives for", mode_name, "computed.\n")

    # 绘制平滑曲线并保存图像
    p <- ggplot() +
      geom_point(data = data, aes(x = interview_age, y = ModeValue), alpha = 0.3) +
      labs(title = paste("Smooth Curve for", mode_name),
           x = "Age",
           y = "Mode Value") +  # 纵坐标名称统一为 Mode Value
      theme_minimal() +
      theme(
        text = element_text(family = "Arial", size = 16),
        panel.grid = element_blank(),
        axis.line = element_line(color = "black"),
        plot.background = element_rect(fill = "white", color = NA),  # 设置背景为白色
        panel.background = element_rect(fill = "white", color = NA)  # 设置面板背景为白色
      )

    # 保存图像到图像输出文件夹
    ggsave(filename = file.path(image_output_dir, paste0(mode_name, "_smooth_curve.png")), plot = p, width = 8, height = 6)

    # 保存原始数据到 CSV
    write.csv(data, file = file.path(csv_output_dir, paste0(mode_name, "_scatter_data.csv")), row.names = FALSE)

    # 保存平滑数据到 CSV
    smoothed_data <- data.frame(interview_age = seq(min(data$interview_age), max(data$interview_age), length.out = increments),
                                ModeValue = predict(gam_model, newdata = data.frame(interview_age = seq(min(data$interview_age), max(data$interview_age), length.out = increments),
                                                                                      sex = rep(median(df_fdc$sex), increments),
                                                                                      motion = rep(median(df_fdc$motion), increments))),
                                ModeName = mode_name)
    write.csv(smoothed_data, file = file.path(csv_output_dir, paste0(mode_name, "_smooth_data.csv")), row.names = FALSE)

    # 保存后验导数数据到 CSV
    write.csv(posterior.derivs.long, file = file.path(csv_output_dir, paste0(mode_name, "_posterior_derivatives.csv")), row.names = FALSE)

    return(smoothed_data)
  }))

  # 合并所有模式的平滑数据
  write.csv(all_smoothed_data, file = file.path(csv_output_dir, smoothed_output_file), row.names = FALSE)
}


image_output_directory <- "./Trajectory/smoothed_curves_AveAlongMode_L1/final_affinity_paper_1st_ca2_500_der_images"
csv_output_directory <- "./Trajectory/smoothed_curves_AveAlongMode_L1/ffinal_affinity_paper_1st_ca2_500_der_smoothed_800csv"
smoothed_output_file <- "final_affinity_paper_1st_ca2_500_der_smoothed_800.csv"

# # 调用函数保存平滑曲线图像和数据
# save_smoothed_curves_der_posterior(gamModels_age_fdc, Components, image_output_directory, csv_output_directory, smoothed_output_file, df_fdc,draws = 1000,  # 生成后验分布的模拟次数
#                      increments = 168,  # 用于生成平滑曲线的增量点数
#                      smooth_var = "interview_age",  # 平滑变量为 interview_age
#                      knots = 3)

# # 使用gratia包，基于interview_age, sex, 和 motion 生成拟合值
# library(gratia)

# library(gratia)

# # 提取所有需要的变量并创建新的数据框 (避免使用 newdata 这个名字)
# prediction_data <- data.frame(
#   interview_age = df_fdc$interview_age,
#   sex = df_fdc$sex,
#   motion = df_fdc$motion
# )

# fitted_values_list <- lapply(seq_along(gamModels_age_fdc), function(i) {
#   model <- gamModels_age_fdc[[i]]
  
#   # 使用 gratia 包的 fitted_values 函数生成拟合值
#   fitted_vals <- fitted_values(model, data = prediction_data)
  
#   # 验证生成的 fitted_vals 数据框是否包含 .fitted 列
#   if (!".fitted" %in% names(fitted_vals)) {
#     stop(paste("Fitted values data frame does not contain '.fitted' column for model:", Components[i]))
#   }
  
#   # 打印部分拟合值数据以进行验证
#   # print(paste("Preview of fitted values for", Components[i], ":"))
#   # print(head(fitted_vals))
  
#   # 提取拟合值列，并命名为对应的 Mode 名称
#   fitted_mode_values <- fitted_vals$.fitted
#   return(fitted_mode_values)
# })


# # 将所有 Mode 的拟合值组合成一个数据框
# fitted_data <- do.call(cbind, fitted_values_list)
# fitted_data <- as.data.frame(fitted_data)

# # 给每个拟合值的列命名，分别对应Mode1到Mode360
# colnames(fitted_data) <- paste0("Mode", 1:360)

# # 计算残差：从原始数据中减去拟合数据
# residuals_data <- df_fdc[, paste0("Mode", 1:360)] - fitted_data[, paste0("Mode", 1:360)]

# # 替换原始数据中的Mode1到Mode360列
# df_fdc[, paste0("Mode", 1:360)] <- residuals_data

# # 将修改后的数据保存为新的CSV文件
# write.csv(df_fdc, file = "E:\\hcpd\\corrupt\\final_peaks_paper_1st_10dis_gripwithcogandsense_residuals_800.csv", row.names = FALSE)
