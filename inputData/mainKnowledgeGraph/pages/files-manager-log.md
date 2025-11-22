source:: [[files-manager]]
icon:: 

- query-table:: true
  #+BEGIN_QUERY
  {:title "All files operate log"
   :query [:find (pull ?b [*])
           :in $ ?current-page
           :where
           [?p :block/name ?current-page]
           [?b :block/page ?p]
           [?b :block/content ?content]
           [(!= ?content "")]
          ]
   :inputs ["files-manager"]}
  #+END_QUERY