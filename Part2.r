popDF=read.table('/Users/rasam/Google Drive/Spring17/PSWD/PSW_mid_term/Population/Population.csv',header=T,sep=',')
#--------------ANSWER_01------------------------------------------
#Ref:http://stackoverflow.com/questions/3445590/how-to-extract-a-subset-of-a-data-frame-based-on-a-condition-involving-a-field
states <- subset(popDF,(popDF$SUMLEV == '40'))
#https://www.r-bloggers.com/r-sorting-a-data-frame-by-the-contents-of-a-column/
#Year 2010
max_2010 =states[order(-states$INTERNATIONALMIG2010)[1:5],]
library(ggplot2)
#Ref:http://docs.ggplot2.org/0.9.3.1/geom_bar.html
ggplot(max_2010, aes(x = max_2010$NAME, y = max_2010$INTERNATIONALMIG2010)) +
  geom_bar(stat = "identity",fill='black') + xlab("States") +
  ylab("No. of International Immigrants") +
  ggtitle("2010's top 5 states with International Immigrants")
#Year 2012
max_2012 =states[order(-states$INTERNATIONALMIG2012)[1:5],]
ggplot(max_2012, aes(x = max_2012$NAME, y = max_2012$INTERNATIONALMIG2012)) +
  geom_bar(stat = "identity",fill='black') + xlab("States") +
  ylab("No. of International Immigrants") +
  ggtitle("2012's top 5 states with International Immigrants")
#Year 2014
max_2014 =states[order(-states$INTERNATIONALMIG2014)[1:5],]
ggplot(max_2014, aes(x = max_2014$NAME, y = max_2014$INTERNATIONALMIG2014)) +
  geom_bar(stat = "identity",fill='black') + xlab("States") +
  ylab("No. of International Immigrants") +
  ggtitle("2014's top 5 states with International Immigrants")

#--------------ANSWER_02------------------------------------------
name(popDF)
#For Region
#Ref:http://stackoverflow.com/questions/10085806/extracting-specific-columns-from-a-data-frame
region_DF = popDF[,c("REGION","POPESTIMATE2010","POPESTIMATE2011","POPESTIMATE2012","POPESTIMATE2013","POPESTIMATE2014")]
#Removing rows with irrelevant data like region '0' and region 'X'
region_DF=subset(region_DF,(region_DF$REGION != 'X'& region_DF$REGION != 0))
#Ref:http://stackoverflow.com/questions/18799901/data-frame-group-by-column
regional_pop = aggregate(. ~ REGION, region_DF, sum)
library(reshape2)
traff2 <- melt(regional_pop,id=c("REGION"),variable.name = "Year")
#Remove the X in the Year column and convert it to number
traff2$Year <- as.numeric(gsub(pattern="POPESTIMATE",replacement = "",x = as.character(traff2$Year)))
options(scipen=10)
ggplot(traff2, aes(x = Year, y = value, color = REGION))+
  facet_grid(facets = REGION~., scales = "free_y")+
  geom_line()+theme_bw()+ylab('Population')

#For Division
div_DF=popDF[,c("DIVISION","POPESTIMATE2010","POPESTIMATE2011","POPESTIMATE2012",
                       "POPESTIMATE2013","POPESTIMATE2014")]
#Removing rows with irrelevant data like region '0' and region 'X'
div_DF=subset(div_DF,(div_DF$DIVISION!=0 & div_DF$DIVISION!='X'))
#Group population by Division
div_pop=aggregate(. ~ DIVISION, div_DF, sum)
traff22 <- melt(div_pop,id=c("DIVISION"),variable.name = "Year")

#Remove the X in the Year column and convert it to number
traff22$Year <- as.numeric(gsub(pattern="POPESTIMATE",replacement = "",x = as.character(traff22$Year)))
options(scipen=10)
ggplot(traff22, aes(x = Year, y = value, color = DIVISION))+
  facet_grid(facets = DIVISION~., scales = "free_y")+
  geom_line()+theme_bw()+ylab('Population')
#Ref:http://stackoverflow.com/questions/27382649/a-line-graph-for-each-row

#--------------ANSWER_03------------------------------------------
x = popDF[,c("DIVISION","NPOPCHG_2012","NPOPCHG_2014")]
x=subset(x,(x$DIVISION!=0 & x$DIVISION!='X'))
#Group population by Division
x_div_pop=aggregate(. ~ DIVISION, x, sum)

#x_div_pop['DIVISION'][which.max(x_div_pop$NPOPCHG_2012)]
#Ref:http://gis.stackexchange.com/questions/97310/return-column-number-of-min-value-in-dataframe
max_pop_rate_2012 = x_div_pop$DIVISION[x_div_pop$NPOPCHG_2012 == max(x_div_pop$NPOPCHG_2012)]
max_pop_rate_2012 = as.character(max_pop_rate_2012)
cat("Division that show the highest increasing rate of population between 2011 and 2012 is: ", max_pop_rate_2012)

max_pop_rate_2014 = x_div_pop$DIVISION[x_div_pop$NPOPCHG_2014 == max(x_div_pop$NPOPCHG_2014)]
max_pop_rate_2014 = as.character(max_pop_rate_2014)
cat("Division that show the highest increasing rate of population between 2013 and 2014 is: ", max_pop_rate_2014)

