df0 <- input1 %>% group_by(equipment) %>%
              summarise(n=n()) %>%
              mutate(n=n/sum(n)*100) %>%
              top_n(10,n) %>%
              arrange(desc(n))