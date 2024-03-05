# LLM Foundations (early book draft)

### See the book live at [https://llms.danwil.com](https://llms.danwil.com).

by Dan Wilhelm

In this book, we develop a low-level understanding of Large Language Models (LLMs). To understand large circuits, we apply learnings from mathematics and the analysis of small circuits.

This book has an accompanying [YouTube channel](https://www.youtube.com/channel/UCS5ef1WKtxYohi_K_Ucmi7A) and project GitHub repo (coming soon).

To get the most out of this book, we recommend the reader be fluent in Python and know the basics of NumPy, linear algebra, and machine learning. Therefore, our target audience includes researchers, CS students, and software engineers with an interest in LLMs.

Currently, our focus is on analysis rather than training.

---
## Building the site

This site is built using the Rust static-site generator [mdBook](https://github.com/rust-lang/mdBook).

To build the site:

1. [Install mdBook](https://rust-lang.github.io/mdBook/guide/installation.html).
2. Install the [katex preprocessor](https://github.com/lzanini/mdbook-katex): `cargo install mdbook-katex`.
3. Install the [mermaid preprocessor](https://github.com/badboy/mdbook-mermaid): `cargo install mdbook-mermaid`.
4. From the project directory, run `mdbook build`. The output will be in the `book` directory.
5. Alternatively, to view the site immediately in watch mode run `mdbook serve --open`.


_Note:_ Also uses a [table-of-contents modification](https://github.com/JorelAli/mdBook-pagetoc) that does not require installation.
