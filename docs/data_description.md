# Opinion Articles Spreadsheet
Contains 9 columns:
- `article_id` - article identifier
- `title` - title of the article
- `authors` - authors of the article
- `body` - full text contained in the article
- `meta_description` - 
- `topics` - topic of the article (genre)
- `keywords` - keywords for the article of possibly important tokens
- `publish_date` - date when article was published
- `url_canonical` - URL for the article

# Opinion Articles ADUs Spreadsheet
Contains 6 columns:
- `article_id` - identifier of article in the other spreadsheet
- `annotator` - what annotator annotated that ADU
- `node` - represents the number of node in the annotator, possibly not relevant
- `ranges` - span where tokens occur in the document (character range)
- `tokens` - tokens being used in the ADU
- `label` - target column, type of ADU