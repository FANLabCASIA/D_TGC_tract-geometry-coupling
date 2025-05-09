################################################## longitudinal rain cloud ##################################################
# install.packages(c("ggplot2", "ggrepel", "ggridges"))
library(ggplot2)
library(ggrepel)
library(ggridges)
library(ggdist)
# install.packages(c("ggbeeswarm"))
library(ggbeeswarm)
# install.packages("ggdist")
library(ggsignif)
library(dplyr)
library(ggpubr)
library(gghalves)

# 创建数据集的示例
# data <- data.frame(
#   subject_id = rep(1:100, each = 2),
#   Age = rep(c("BL", "FU2"), 100),
#   TGC = c(rnorm(100, 3, 0.5), rnorm(100, 3.2, 0.5)),
#   group = rep(c("group1"), each = 100)
# )
# print(data)

# 对于一个纤维束的不同mode的遍历
# fiber_name_ = "UF"
# for (iii in 1:199) {
# name = paste0(fiber_name_, "_Mode",iii)
# data <- read.csv(paste0("/Users/deyingli/Documents/code/R_dir/raincloud/data/",fiber_name_,"/",name,"_bl-fu2-fu3.csv"))

# 对于所有纤维束的遍历
file_path <- "/Users/deyingli/Documents/code/R_dir/raincloud/data/fiber_name_ori_nonum_nohemi.txt"  # 替换为你的文件路径
lines <- readLines(file_path)
for (fiber_name_ in lines) {
name = paste0(fiber_name_, "_L1_norm")
print(name)
data <- read.csv(paste0("/Users/deyingli/Documents/code/R_dir/raincloud/data/L1_norm/",fiber_name_,"_L1_norm_bl-fu2-fu3.csv"))

# 创建图形
Custom.color <- c("#245892", "#877eac","#a9d296")
p <- ggplot(data, aes(x = Age, y = TGC, group = subject_id)) + 
            geom_line(aes(group = subject_id), alpha = 0.3, color='#bbbbb9') +  # 添加连接线
            geom_beeswarm(aes(color = Age), size = 2, alpha = 0.6) +  # 随机扰动散点图
            # geom_beeswarm(aes(color = Age), size = 2, alpha = 0.6, groupOnX = TRUE, dodge.width = 0.2) +  # 瘦一点的随机扰动散点图
            
            # 小提琴图
            geom_half_violin(data = subset(data, Age == "BL"), 
                        aes(x = as.numeric(as.factor(Age)) - 0.15, y = TGC, fill = Age,group=Age),
                        side='l',width = 0.4, justification = 1, point_colour = NA, alpha = 0.6) + 
            geom_half_violin(data = subset(data, Age == "FU2"), 
                        aes(x = as.numeric(as.factor(Age)) + 1.15, y = TGC, fill = Age,group=Age),
                        side='r',width = 0.4, justification = -1, point_colour = NA, alpha = 0.6) + 
            geom_half_violin(data = subset(data, Age == "FU3"), 
                        aes(x = as.numeric(as.factor(Age)) + 2.15, y = TGC, fill = Age,group=Age),
                        side='r',width = 0.4, justification = -1, point_colour = NA, alpha = 0.6) + 

            # 箱线图
            geom_boxplot(data = subset(data, Age == "BL"), 
                        aes(x = as.numeric(as.factor(Age)) - 0.15, y = TGC, fill = Age,group=Age),
                        width = 0.1, outlier.size = 0, outlier.alpha = 0, alpha = 0.5) + 
            geom_boxplot(data = subset(data, Age == "FU2"), 
                        aes(x = as.numeric(as.factor(Age)) + 1.15, y = TGC, fill = Age,group=Age),
                        width = 0.1, outlier.size = 0, outlier.alpha = 0, alpha = 0.5) + 
            geom_boxplot(data = subset(data, Age == "FU3"), 
                        aes(x = as.numeric(as.factor(Age)) + 2.15, y = TGC, fill = Age,group=Age),
                        width = 0.1, outlier.size = 0, outlier.alpha = 0, alpha = 0.5) + 

            # 添加配对t检验结果
            stat_compare_means(method = "t.test", paired = TRUE, label = "p.format", 
                               comparisons = list(c("BL", "FU2"), c("FU2", "FU3"), c("BL", "FU3"))) + 
            
            # color
            scale_fill_manual(values = c("BL" = "#e8b7c3", "FU2" = "#7484a6", "FU3" = "#a9d296")) +  # 自定义颜色
            scale_color_manual(values = c("BL" = "#e8b7c3", "FU2" = "#7484a6", "FU3" = "#a9d296")) +  # 自定义颜色
            # scale_color_manual(values=Custom.color) + 
            # scale_fill_manual(values=Custom.color) + 

            # theme_minimal() +  # 使用简洁主题
            theme(axis.ticks.x=element_line(size=0,color="black"),
                  panel.background=element_rect(fill="white",color="white"),
                  panel.grid.major.x=element_blank(),
                  panel.grid.minor.x=element_blank(),
                  panel.grid.major.y=element_line(color="white",size=0.25),
                  panel.grid.minor.y=element_blank(),
                  panel.border=element_rect(color="black",fill=NA,linewidth=1),
                  legend.position="none",
                  axis.title.x=element_text(size=13),
                  # axis.title.y=element_text(size=13),
                  axis.title.y=element_blank(),
                  axis.text.x=element_text(size=12,hjust=0.3),
                  # axis.text.y=element_text(size=12),
                  axis.text.y=element_blank(),
                  plot.title=element_text(hjust=0.5)
                  ) + 
            labs(title = "IMAGEN validation", x = "", y = "TGC value")  # 添加标签

# 显示图形
png(file = paste0("./raincloud/figures/",name,".png"), width = 1400, height = 1800, units = 'px', res = 300)
# pdf(file = paste0("./raincloud/figures/",name,".pdf"), width = 1400, height = 1800)
print(p)
dev.off()
}