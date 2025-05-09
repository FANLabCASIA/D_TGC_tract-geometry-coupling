# 安装并加载必要的包
# install.packages("dplyr")
# install.packages("ggplot2")
# install.packages("MetBrewer")
# install.packages("colorspace")
library(dplyr)
library(ggplot2)
library(MetBrewer)
library(colorspace)

# load data
fiber_name = 'T_PREF'
corr_true = 0.95
nets_with_weights <- read.csv(paste0("/Users/deyingli/Documents/code/R_dir/plotting/data_ForAZ/",fiber_name,"_2SurWay_SurCorrValue.csv"))

# 定义颜色
isfahan <- MetBrewer::met.brewer("Isfahan1")

# 绘制不同组不同颜色的直方图
p <- ggplot(data = nets_with_weights, aes(x = corr, fill = factor(exp))) + 
  geom_histogram(bins = 50, color = "white") +
  scale_fill_manual(values = colorspace::lighten(isfahan, 0.35)) +
  labs(title = fiber_name, x = "Accuracy (Null models)", y = "Count", fill = "exp") +

  # theme_minimal() +
  theme(panel.grid.minor = element_blank(),
        plot.background = element_rect(fill = "white", color = NA),
        plot.title = element_text(face = "bold"),
        axis.title = element_text(face = "bold"),
        strip.text = element_text(face = "bold", size = rel(0.8), hjust = 0),
        strip.background = element_rect(fill = "grey80", color = NA),
        legend.title = element_text(face = "bold")) +
  
  # true accuracy
  geom_vline(xintercept = corr_true, linetype = "dashed", color = "black", size = 0.5) +
  geom_segment(aes(x = corr_true-0.1, y = 130, xend = corr_true, yend = 150),
               arrow = arrow(length = unit(0.2, "cm")), color = "black") +
  annotate("text", x = corr_true-0.2, y = 100, label = paste0("Accuracy (True)\nr = ",corr_true), vjust = -0.2, color = "black")


# 显示图形
png(file = paste0('./plotting/data_ForAZ/',fiber_name,'_SurCorrHist.png'), width = 2000, height = 1200, units = 'px', res = 300)
print(p)
dev.off()
