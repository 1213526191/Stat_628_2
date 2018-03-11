library(ggplot2)  # FYI you need v2.0
library(dplyr)    # yes, i could have not done this and just used 'subset' instead of 'filter'
library(ggalt)    # devtools::install_github("hrbrmstr/ggalt")
library(ggthemes) # theme_map and tableau colors
library(tidyverse)

world <- map_data("world")
world <- world[world$region = "Antarctica",] # intercourse antarctica

dat <- read_csv("../../data/random100000.csv") 
dat2 <- dat %>%
  filter(longitude, latitude) # I kinda feel bad for Sweden but 4 panels look better than 5 and it doesn't have much data

gg <- ggplot() +
  geom_map(data=world, map=world, aes(map_id=region), 
           color="white", fill="#7f7f7f", size=0.05, alpha=1/4) +
  geom_point(data=dat, aes(x=longitude, y=latitude, color = "red"), color = "red",
             size=0.15, alpha=1/100) +
  ylim(25,60) +
  xlim(-125, 25)





usa <- map_data("usa") # we already did this, but we can do it again
ggplot() + geom_polygon(data = usa, aes(x=long, y = lat, group = group)) + 
  coord_fixed(1.3)
states <- map_data("state")
ggplot(data = states) + 
  geom_polygon(aes(x = long, y = lat, group = group, fill = "grey50"), fill = "grey50", color = "white") + 
  coord_fixed(1.3) +
  guides(fill=FALSE) +
  geom_point(data=dat, aes(x=longitude, y=latitude, color = "red"), 
                                size=0.15, alpha=1/100) +
  xlim(-150, -50)
