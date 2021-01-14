df0 <- input1 %>% mutate(time_between = trending_date - publish_time) %>%
     group_by(time_between) %>%
     summarize(Sum=n()) %>%
     arrange(time_between) %>%
     head(10)