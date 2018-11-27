#!/usr/bin/env Rscript

library(readr)
library(ggpubr)

args = commandArgs(trailingOnly=TRUE)

outFile = args[1]

datafile1 <- read_delim('umap.txt',"\t", escape_double = FALSE, col_names = FALSE,trim_ws = TRUE)

datafile1$CellTypes=datafile1$X3

plot1 = ggscatter(datafile1,'X1','X2',color='CellTypes',xlab = "Coordinate 1", ylab = "Coordinate 2", size=2, palette = c('#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00','#ffff33'))+ theme(legend.position = "right")

plot1 = plot1 + ggtitle("UMAP")

datafile2 <- read_delim('tfidfSVD.txt',"\t", escape_double = FALSE, col_names = FALSE,trim_ws = TRUE)

datafile2$CellTypes=datafile2$X3

plot2 = ggscatter(datafile2,'X1','X2',color='CellTypes',xlab = "Coordinate 1", ylab = "Coordinate 2", size=2, palette = c('#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00','#ffff33'))+ theme(legend.position = "right")

plot2 = plot2 + ggtitle("tfidfSVD")

datafile3 <- read_delim('nmf.txt',"\t", escape_double = FALSE, col_names = FALSE,trim_ws = TRUE)

datafile3$CellTypes=datafile3$X3

plot3 = ggscatter(datafile3,'X1','X2',color='CellTypes',xlab = "Coordinate 1", ylab = "Coordinate 2", size=2, palette = c('#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00','#ffff33'))+ theme(legend.position = "right")

plot3 = plot3 + ggtitle("NMF")

datafile4 <- read_delim('justSVD.txt',"\t", escape_double = FALSE, col_names = FALSE,trim_ws = TRUE)

datafile4$CellTypes=datafile4$X3

plot4 = ggscatter(datafile4,'X1','X2',color='CellTypes',xlab = "Coordinate 1", ylab = "Coordinate 2", size=2, palette = c('#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00','#ffff33'))+ theme(legend.position = "right")

plot4 = plot4 + ggtitle("justSVD")

multiplot <- function(..., plotList=NULL, File, cols=1, layout=NULL) {
        library(grid)
        plots <- c(list(...), plotList)
        numPlots = length(plots)
        if (is.null(layout)) {
                layout <- matrix(seq(1, cols * ceiling(numPlots/cols)),
                                                ncol = cols, nrow = ceiling(numPlots/cols))
        }
        if (numPlots==1) {
                print(plots[[1]])
        } else {
                grid.newpage()
                pushViewport(viewport(layout = grid.layout(nrow(layout), ncol(layout))))
                # Make each plot, in the correct location
                for (i in 1:numPlots) {
                        matchidx <- as.data.frame(which(layout == i, arr.ind = TRUE))
                        print(plots[[i]], vp = viewport(layout.pos.row = matchidx$row,
                                                                                  layout.pos.col = matchidx$col))
                }
        }
}

png(outFile,width=800,height=400)

multiplot(plot1,plot2,plot3,plot4,cols=2)

dev.off()