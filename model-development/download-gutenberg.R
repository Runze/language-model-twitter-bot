library(gutenbergr)
library(tidytext)
library(tidyverse)
options(stringsAsFactors = F)

authors_to_download = c(
  'Dickens, Charles',
  'Austen, Jane',
  'Doyle, Arthur Conan',
  'Wilde, Oscar',
  'Carroll, Lewis',
  'Brontë, Charlotte',
  'Brontë, Emily',
  'Fitzgerald, F. Scott (Francis Scott)'
)

all_works = gutenberg_works()

works_to_download = all_works %>%
  filter(has_text & !is.na(title) & language == 'en') %>%
  filter(author %in% authors_to_download)

# Download
works = gutenberg_download(unique(works_to_download$gutenberg_id), meta_fields = c('title', 'author'))

# Clean punctuations
works = works %>%
  mutate(text = gsub("‘|’", "'", text),
         text = gsub("“|”", '"', text))

write_csv(works, 'data/works.csv')
