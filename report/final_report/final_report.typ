// Some definitions presupposed by pandoc's typst output.
#let blockquote(body) = [
  #set text( size: 0.92em )
  #block(inset: (left: 1.5em, top: 0.2em, bottom: 0.2em))[#body]
]

#let horizontalrule = line(start: (25%,0%), end: (75%,0%))

#let endnote(num, contents) = [
  #stack(dir: ltr, spacing: 3pt, super[#num], contents)
]

#show terms: it => {
  it.children
    .map(child => [
      #strong[#child.term]
      #block(inset: (left: 1.5em, top: -0.4em))[#child.description]
      ])
    .join()
}

// Some quarto-specific definitions.

#show raw.where(block: true): set block(
    fill: luma(230),
    width: 100%,
    inset: 8pt,
    radius: 2pt
  )

#let block_with_new_content(old_block, new_content) = {
  let d = (:)
  let fields = old_block.fields()
  fields.remove("body")
  if fields.at("below", default: none) != none {
    // TODO: this is a hack because below is a "synthesized element"
    // according to the experts in the typst discord...
    fields.below = fields.below.abs
  }
  return block.with(..fields)(new_content)
}

#let empty(v) = {
  if type(v) == str {
    // two dollar signs here because we're technically inside
    // a Pandoc template :grimace:
    v.matches(regex("^\\s*$")).at(0, default: none) != none
  } else if type(v) == content {
    if v.at("text", default: none) != none {
      return empty(v.text)
    }
    for child in v.at("children", default: ()) {
      if not empty(child) {
        return false
      }
    }
    return true
  }

}

// Subfloats
// This is a technique that we adapted from https://github.com/tingerrr/subpar/
#let quartosubfloatcounter = counter("quartosubfloatcounter")

#let quarto_super(
  kind: str,
  caption: none,
  label: none,
  supplement: str,
  position: none,
  subrefnumbering: "1a",
  subcapnumbering: "(a)",
  body,
) = {
  context {
    let figcounter = counter(figure.where(kind: kind))
    let n-super = figcounter.get().first() + 1
    set figure.caption(position: position)
    [#figure(
      kind: kind,
      supplement: supplement,
      caption: caption,
      {
        show figure.where(kind: kind): set figure(numbering: _ => numbering(subrefnumbering, n-super, quartosubfloatcounter.get().first() + 1))
        show figure.where(kind: kind): set figure.caption(position: position)

        show figure: it => {
          let num = numbering(subcapnumbering, n-super, quartosubfloatcounter.get().first() + 1)
          show figure.caption: it => {
            num.slice(2) // I don't understand why the numbering contains output that it really shouldn't, but this fixes it shrug?
            [ ]
            it.body
          }

          quartosubfloatcounter.step()
          it
          counter(figure.where(kind: it.kind)).update(n => n - 1)
        }

        quartosubfloatcounter.update(0)
        body
      }
    )#label]
  }
}

// callout rendering
// this is a figure show rule because callouts are crossreferenceable
#show figure: it => {
  if type(it.kind) != str {
    return it
  }
  let kind_match = it.kind.matches(regex("^quarto-callout-(.*)")).at(0, default: none)
  if kind_match == none {
    return it
  }
  let kind = kind_match.captures.at(0, default: "other")
  kind = upper(kind.first()) + kind.slice(1)
  // now we pull apart the callout and reassemble it with the crossref name and counter

  // when we cleanup pandoc's emitted code to avoid spaces this will have to change
  let old_callout = it.body.children.at(1).body.children.at(1)
  let old_title_block = old_callout.body.children.at(0)
  let old_title = old_title_block.body.body.children.at(2)

  // TODO use custom separator if available
  let new_title = if empty(old_title) {
    [#kind #it.counter.display()]
  } else {
    [#kind #it.counter.display(): #old_title]
  }

  let new_title_block = block_with_new_content(
    old_title_block, 
    block_with_new_content(
      old_title_block.body, 
      old_title_block.body.body.children.at(0) +
      old_title_block.body.body.children.at(1) +
      new_title))

  block_with_new_content(old_callout,
    block(below: 0pt, new_title_block) +
    old_callout.body.children.at(1))
}

// 2023-10-09: #fa-icon("fa-info") is not working, so we'll eval "#fa-info()" instead
#let callout(body: [], title: "Callout", background_color: rgb("#dddddd"), icon: none, icon_color: black, body_background_color: white) = {
  block(
    breakable: false, 
    fill: background_color, 
    stroke: (paint: icon_color, thickness: 0.5pt, cap: "round"), 
    width: 100%, 
    radius: 2pt,
    block(
      inset: 1pt,
      width: 100%, 
      below: 0pt, 
      block(
        fill: background_color, 
        width: 100%, 
        inset: 8pt)[#text(icon_color, weight: 900)[#icon] #title]) +
      if(body != []){
        block(
          inset: 1pt, 
          width: 100%, 
          block(fill: body_background_color, width: 100%, inset: 8pt, body))
      }
    )
}



#let article(
  title: none,
  subtitle: none,
  authors: none,
  date: none,
  abstract: none,
  abstract-title: none,
  cols: 1,
  margin: (x: 1.25in, y: 1.25in),
  paper: "us-letter",
  lang: "en",
  region: "US",
  font: "libertinus serif",
  fontsize: 11pt,
  title-size: 1.5em,
  subtitle-size: 1.25em,
  heading-family: "libertinus serif",
  heading-weight: "bold",
  heading-style: "normal",
  heading-color: black,
  heading-line-height: 0.65em,
  sectionnumbering: none,
  pagenumbering: "1",
  toc: false,
  toc_title: none,
  toc_depth: none,
  toc_indent: 1.5em,
  doc,
) = {
  set page(
    paper: paper,
    margin: margin,
    numbering: pagenumbering,
  )
  set par(justify: true)
  set text(lang: lang,
           region: region,
           font: font,
           size: fontsize)
  set heading(numbering: sectionnumbering)
  if title != none {
    align(center)[#block(inset: 2em)[
      #set par(leading: heading-line-height)
      #if (heading-family != none or heading-weight != "bold" or heading-style != "normal"
           or heading-color != black or heading-decoration == "underline"
           or heading-background-color != none) {
        set text(font: heading-family, weight: heading-weight, style: heading-style, fill: heading-color)
        text(size: title-size)[#title]
        if subtitle != none {
          parbreak()
          text(size: subtitle-size)[#subtitle]
        }
      } else {
        text(weight: "bold", size: title-size)[#title]
        if subtitle != none {
          parbreak()
          text(weight: "bold", size: subtitle-size)[#subtitle]
        }
      }
    ]]
  }

  if authors != none {
    let count = authors.len()
    let ncols = calc.min(count, 3)
    grid(
      columns: (1fr,) * ncols,
      row-gutter: 1.5em,
      ..authors.map(author =>
          align(center)[
            #author.name \
            #author.affiliation \
            #author.email
          ]
      )
    )
  }

  if date != none {
    align(center)[#block(inset: 1em)[
      #date
    ]]
  }

  if abstract != none {
    block(inset: 2em)[
    #text(weight: "semibold")[#abstract-title] #h(1em) #abstract
    ]
  }

  if toc {
    let title = if toc_title == none {
      auto
    } else {
      toc_title
    }
    block(above: 0em, below: 2em)[
    #outline(
      title: toc_title,
      depth: toc_depth,
      indent: toc_indent
    );
    ]
  }

  if cols == 1 {
    doc
  } else {
    columns(cols, doc)
  }
}

#set table(
  inset: 6pt,
  stroke: none
)

#show: doc => article(
  title: [Sniffing for Phishing],
  subtitle: [Final Report for DSCI 591 Capstone Project],
  authors: (
    ( name: [Ci Xu],
      affiliation: [],
      email: [] ),
    ( name: [Danish Karlin Isa],
      affiliation: [],
      email: [] ),
    ( name: [Jia Quan (Joseph) Lim],
      affiliation: [],
      email: [] ),
    ( name: [Lik Hang (Alex) Wong],
      affiliation: [],
      email: [] ),
    ),
  date: [June 25, 2025],
  sectionnumbering: "1.1.a",
  pagenumbering: "1",
  toc: true,
  toc_title: [Table of contents],
  toc_depth: 3,
  cols: 1,
  doc,
)

= Executive Summary
<executive-summary>
UBC Cybersecurity (the Partner) currently reviews suspicious emails manually, a process hindered by high volume and time sensitivity.

We developed a machine learning (ML) pipeline to automatically classify emails as `benign` or `malicious`. This helps to reduce manual workload, accelerate threat detection, and improve cybersecurity at UBC.

We evaluated classical ML models, BERT-based architectures, and large language models (LLMs). The final solution, PhishSense-v2, uses a stacked ensemble of four parallel `XGBClassifier` models, combined by a `SVC` model.

PhishSense-v2 managed to achieve an F1-score of 0.75 on the `benign` class, marking a 50 percent improvement over the Partner's existing ML pipeline.

= Introduction
<introduction>
Phishing emails pose a major cybersecurity threat, often designed to trick recipients into revealing sensitive information such as credentials and personal data. According to #cite(<internet_crime_complaint_center_internet_2025>, form: "prose");, business email compromise#footnote[Business email compromise is a form of scam that targets users in an organisation and seek information with the intention of defrauding the organisation.] ranks second in financial impact. At UBC, email users are frequent targets of malicious email campaigns and are at risk of engaging with malicious content, putting both individuals and the institution at risk.

Users can report suspicious emails to the Partner for investigation. Emails identified as malicious and posing an urgent threat are recalled to prevent further exposure.

The Partner currently uses a ML pipeline (#ref(<fig-current-pipeline>, supplement: [Figure])) to classify reported emails. However, its performance is limited by its reliance solely on the email's subject and body. As a result, Analysts must manually review all reported emails to determine whether similar messages should be recalled from other inboxes. Given the high volume of reports, this process is both time-consuming and resource intensive.

#figure([
#block[
#box(image("final_report_files/figure-typst/mermaid-figure-1.png", height: 1.2in, width: 6in))

]
], caption: figure.caption(
position: bottom, 
[
Current ML pipeline used by the Partner
]), 
kind: "quarto-float-fig", 
supplement: "Figure", 
)
<fig-current-pipeline>


== Objectives
<objectives>
In collaboration with the Partner, we aim to develop a new ML pipeline that classifies an email as either `benign` or `malicious` through:

+ Enhancing feature extraction and engineering to better leverage both email metadata and content.

+ Applying a combination of classical and advanced ML techniques to improve classification accuracy.

The primary objective is to accurately identify benign emails, thereby reducing the number of tickets requiring manual review. This, in turn, shortens the Partner's response time and strengthens cybersecurity at UBC.

= Proposed Data Science Techniques
<proposed-data-science-techniques>
We explored three distinct approaches in this project:

+ Classical ML models, with a strong emphasis on feature engineering to extract meaningful signals from email data.

+ Bidirectional Encoder Representations from Transformers (BERT) language models, which learn complex patterns directly from raw text using deep contextual embeddings.

+ Large language models (LLMs), which leverage pre-trained models and prompt engineering to perform classification with minimal task-specific training.

== Approach 1: Classical ML models
<approach-1-classical-ml-models>
Classical ML models are relatively simple and have low computational overhead. This approach relies heavily on feature engineering to construct a representation of each email that is both information-rich and tailored to the model's predictive capabilities. By transforming raw email data into structured features, we aimed to capture relevant patterns that could distinguish between benign and malicious messages.

=== Feature extraction
<feature-extraction>
We retrieved metadata and content from emails using the Python `email` package. The extracted metadata#footnote[Examples include sender and recipient email addresses, routing information, and authentication results. See #ref(<sec-appendix-1>, supplement: [Section]) for a list of features extracted.] can provide valuable signals for assessing the potential malicious intent of an email.

To ensure consistency and robustness, we only considered header fields that comply with Request For Comments (RFC) standards. Non-compliant header fields#footnote[Non-compliant fields are typically prefixed with `X-` and often appended by receiving mail servers.] were dropped due to their variability and lack of standardisation.

Email content is typically multipart, with each part associated with a content type. Most emails include a body in either HTML or plain text format or both. Many also contain additional content types such as images, videos, or application files. For the purposes of this project, we retained only the email body (both HTML and plain text parts) as the primary source of content for analysis. `CountVectorizer` was used to obtain representations of the plain text email body.

=== Feature engineering
<feature-engineering>
To enhance predictive performance, we engineered features#footnote[See #ref(<sec-appendix-1>, supplement: [Section]) for a list of engineered features.] that capture key characteristics distinguishing benign emails from malicious ones. These features were selected based on domain expertise provided by the Partner and research in phishing and spam detection, with the goal of creating a feature set that is both interpretable and effective for classification.

We incorporated email authentication#footnote[Supported authentication protocols include Sender Policy Framework (SPF), DomainKeys Identified Mail (DKIM), and Domain-based Message Authentication, Reporting, and Conformance (DMARC).] results that verify the legitimacy of the sender's domain and ensure that the message has not been altered in transit, which can be strong indicators of email authenticity.

In addition, we engineered features to detect spoofing or obfuscation attempts, such as the number of mail server hops, mismatches between the `From` and `Reply-To` domains, and discrepancies between the name servers of the originating mail server and the sender's domain.

Malicious emails are also often generated using tools purchased on the dark web, resulting in messages that are poorly written, heavily formatted in HTML, and designed to evade detection. To capture these traits, we included features such as the number of grammatical errors, the frequency of whitespace#footnote[Whitespace characters include spaces, tabs and newline characters.] characters, and the presence and types of non-textual content#footnote[Non-texual content consist multimedia and application files.];.

=== Classifiers
<classifiers>
#ref(<tbl-classical-models>, supplement: [Table]) lists the classical models evaluated in this project. These models were chosen based on their interpretability, computational efficiency, and historical effectiveness in text and email classification tasks.

#figure([
#table(
  columns: (35%, 65%),
  align: (left,left,),
  table.header([Model], [Description],),
  table.hline(),
  [`DummyClassifier` @sklearn_api], [Serves as a baseline for benchmarking model performance],
  [`LogisticRegression` @sklearn_api], [Interpretable, captures linear relationships between features and target label @hilbe2016practical],
  [`GaussianNB` (Naïve Bayes) @sklearn_api], [Simple model well-suited for text classification, efficient training time @sammut2011encyclopedia],
  [`SVC` (Support Vector Machine) @sklearn_api], [Effective for binary classification in high-dimensional spaces @sammut2011encyclopedia],
  [`RandomForestClassifier` @sklearn_api], [Robust to noisy data @parmar2018review],
  [`XGBClassifier` @xgboost], [Captures non-linear relationships, supports regularization, generalizes well @omotehinwa2023hyperparameter],
)
], caption: figure.caption(
position: top, 
[
Classical models evaluated in this project with their associated rationales
]), 
kind: "quarto-float-tbl", 
supplement: "Table", 
)
<tbl-classical-models>


We also explored an ensemble learning approach using `StackingClassifier` @sklearn_api, where multiple base models were trained in parallel to generate predictions. These predictions were then used as inputs for a final estimator, which produced the overall prediction. An example of the architecture for this ensemble method is illustrated in #ref(<fig-stacking>, supplement: [Figure]).

#figure([
#block[
#box(image("final_report_files/figure-typst/mermaid-figure-5.png", height: 3.27in, width: 3in))

]
], caption: figure.caption(
position: bottom, 
[
Example architecture of a StackingClassifier
]), 
kind: "quarto-float-fig", 
supplement: "Figure", 
)
<fig-stacking>


== Approach 2: BERT language models
<approach-2-bert-language-models>
BERT language models captures rich contextual information by modelling the semantics and relationships between words. It is highly effective for natural language processing (NLP) tasks such as classification, topic modelling, and sentiment analysis. BERT is particularly well-suited for this project, as phishing indicators are often embedded in the nuanced language of email bodies.

We applied open-source models to the cleaned plain-text content of emails to classify emails as `benign` or `malicious`, and generate labels as part of feature engineering to improve the performance of classical ML models.

#ref(<tbl-bert-models>, supplement: [Table]) lists the BERT models evaluated in this project.

#figure([
#table(
  columns: (18.18%, 45.45%, 36.36%),
  align: (left,left,left,),
  table.header([Task], [Hugging Face repository], [Description],),
  table.hline(),
  [Classification], [ElSlay/BERT-Phishing-Email-Model @ElSlay], [Trained on 18,600 emails labelled as either `malicious` or `benign`],
  [Classification], [ealvaradob/bert-finetuned-phishing @ealvaradob], [Trained on 80,000 emails, text messages, URLs, and HTML content labelled as either `malicious` or `benign`],
  [Feature Engineering (Topic Modelling)], [MaartenGr/BERTopic\_Wikipedia @MaartenGr], [Trained on 1M+ Wikipedia pages; generates \~2,300 topic labels],
  [Feature Engineering (Sentiment Analysis)], [cardiffnlp/twitter-roberta-base-sentiment-latest @camacho-collados-etal-2022-tweetnlp], [Trained on 124M tweets; classifies text as `positive`, `neutral`, or `negative`],
)
], caption: figure.caption(
position: top, 
[
BERT language models evaluated in this project
]), 
kind: "quarto-float-tbl", 
supplement: "Table", 
)
<tbl-bert-models>


== Approach 3: LLMs
<approach-3-llms>
LLMs are well-suited for this project due to their ability to infer meaning from unstructured text, outperforming models like BERT. Unlike classical models, LLMs require minimal feature engineering.

A locally-hostel LLM#footnote[Microsoft Phi-4-mini reasoning model (3.84B parameters) quantised by Unsloth AI with 4,096 token content window] @UnslothAI was used for zero-shot email classification and to reverse-engineer label assignments, supporting feature engineering for classical models. In both cases, only the email header or body was provided due to context window limitations. The general workflow is illustrated in #ref(<fig-llm-general-workflow>, supplement: [Figure]).

#figure([
#block[
#box(image("final_report_files/figure-typst/mermaid-figure-4.png", height: 5.85in, width: 2.5in))

]
], caption: figure.caption(
position: bottom, 
[
Workflow for LLM
]), 
kind: "quarto-float-fig", 
supplement: "Figure", 
)
<fig-llm-general-workflow>


#ref(<tbl-llm-json>, supplement: [Table]) describes the fields in structured JSON response.

#figure([
#table(
  columns: (18.92%, 35.14%, 45.95%),
  align: (left,left,left,),
  table.header([Field], [Description], [Possible values],),
  table.hline(),
  [Label], [Label assigned by LLM], [`benign`, `malicious`],
  [Confidence], [Confidence level of the label assignment], [`low` (weak/conflicting evidence), `medium` (some uncertainty), `high` (strong evidence)],
  [Reasons], [List containing three reasons justifying given label], [Short justifications referencing parts of the email],
)
], caption: figure.caption(
position: top, 
[
LLM response formatted in JSON
]), 
kind: "quarto-float-tbl", 
supplement: "Table", 
)
<tbl-llm-json>


We leveraged prompt engineering#footnote[Refer to Appendix 1 for a list of prompts used.] to improve the quality and ensure consistency of the LLM's responses.

= Proposed data product: PhishSense-v2
<proposed-data-product-phishsense-v2>
PhishSense-v2 is a containerised ML pipeline that predicts the probability of an email being `benign`. The pipeline is visualised in #ref(<fig-phishsense-v2>, supplement: [Figure]).

#figure([
#block[
#box(image("final_report_files/figure-typst/mermaid-figure-3.png", height: 4.11in, width: 2.5in))

]
], caption: figure.caption(
position: bottom, 
[
Pipeline for PhishSense-v2
]), 
kind: "quarto-float-fig", 
supplement: "Figure", 
)
<fig-phishsense-v2>


It is encapsulated in a Docker container and callable via a RESTful API, allowing PhishSense-v2 to serve as a drop-in replacement for the current PhishSense. This design aligns with the Partner's specifications, requiring minimal workflow changes for Cybersecurity Analysts reviewing reported emails.

The step-by-step workflow is described in #ref(<fig-workflow>, supplement: [Figure]).

#figure([
#block[
#box(image("final_report_files/figure-typst/mermaid-figure-2.png", height: 9.26in, width: 2.5in))

]
], caption: figure.caption(
position: bottom, 
[
Step-by-step workflow
]), 
kind: "quarto-float-fig", 
supplement: "Figure", 
)
<fig-workflow>


= Implementation
<implementation>
== Dataset
<dataset>
The dataset comprises 25,625 emails that were reported as suspicious by users and had already bypassed existing rule-based filters. They are in the `eml` file format and are manually reviewed and labelled by the Partner.

#ref(<tbl-target-definitions>, supplement: [Table]) defines the target labels used by the Partner.

#figure([
#table(
  columns: (12.5%, 31.25%, 56.25%),
  align: (left,left,left,),
  table.header([Target], [Threat level when reviewed], [Sub-labels],),
  table.hline(),
  [`benign`], [Little/no threat], [`legitimate`, `spam`],
  [`malicious`], [High/active threat], [`CEO_fraud`, `phishing`, `reply-chain-attack`],
)
], caption: figure.caption(
position: top, 
[
Definitions of target labels used by the Partner
]), 
kind: "quarto-float-tbl", 
supplement: "Table", 
)
<tbl-target-definitions>


The dataset was split into train (70%) and test (30%) sets to ensure robust model evaluation and prevent data leakage @rosenblatt2024data. To facilitate rapid prototyping and debugging, we created two smaller subsets of the train set containing 3,000 and 200 samples respectively using stratified random sampling @singh1996stratified. This preserves statistical characteristics of the full train set @dobbin2011optimally and ensures that the evaluation metrics reflect the model's performance across all classes, particularly in the presence of class imbalance.

#ref(<fig-eda-emails>, supplement: [Figure]) shows the distribution of emails across the target labels for the train set.

#figure([
#box(image("final_report_files/figure-typst/fig-eda-emails-output-1.png"))
], caption: figure.caption(
position: bottom, 
[
Distribution of emails in train set
]), 
kind: "quarto-float-fig", 
supplement: "Figure", 
)
<fig-eda-emails>


== Evaluation metrics
<evaluation-metrics>
Model performance will be evaluated using the F1-score#footnote[Refer to #ref(<sec-appendix-2>, supplement: [Section]) for definitions.] for the `benign` class and the overall false negative/benign rate#footnote[Refer to #ref(<sec-appendix-2>, supplement: [Section]) for definitions.] (FNR). A high F1-score reflects the model's ability to correctly identify most `benign` emails while minimising false negative/benign predictions. Maintaining a low FNR is also critical to avoid mistakenly dismissing malicious emails, which would increase risk exposure for UBC email users.

The Partner has set performance targets of an F1-score ≥ 0.85 and an FNR ≤ 0.001.

= Results
<results>
== Proposed data science techniques
<proposed-data-science-techniques-1>
=== Approach 1: Classical ML models
<approach-1-classical-ml-models-1>
#ref(<tbl-classical-results>, supplement: [Table]) outlines the cross-validation results of the classical models.

#figure([
#table(
  columns: (24%, 19%, 19%, 19%, 19%),
  align: (left,left,left,left,left,),
  table.header([Model], [Fit time (s)], [Score time (s)], [Train score], [Valid score],),
  table.hline(),
  [DummyClassifier], [0.408 (+/- 0.025)], [0.097 (+/- 0.018)], [0.000 (+/- 0.000)], [0.000 (+/- 0.000)],
  [LogisticRegression], [0.875 (+/- 0.090)], [0.097 (+/- 0.021)], [0.990 (+/- 0.002)], [0.609 (+/- 0.065)],
  [GaussianNB], [1.183 (+/- 0.093)], [0.295 (+/- 0.009)], [0.969 (+/- 0.004)], [0.561 (+/- 0.066)],
  [SVC], [1.237 (+/- 0.039)], [0.278 (+/- 0.021)], [0.483 (+/- 0.018)], [0.281 (+/- 0.120)],
  [RandomForestClassifier], [2.461 (+/- 0.265)], [0.124 (+/- 0.015)], [0.999 (+/- 0.001)], [0.493 (+/- 0.105)],
  [XGBClassifier], [4.098 (+/- 0.256)], [0.102 (+/- 0.018)], [0.998 (+/- 0.001)], [0.605 (+/- 0.071)],
)
], caption: figure.caption(
position: top, 
[
Cross-validation results for classical models
]), 
kind: "quarto-float-tbl", 
supplement: "Table", 
)
<tbl-classical-results>


We also implemented an ensemble model using `StackingClassifier`, where two separate `XGBClassifier` models are trained on header and content features, respectively. Their predictions are combined by a final estimator to produce the overall output. The performance of this ensemble approach is summarized in #ref(<tbl-stacking-results>, supplement: [Table]).

#figure([
#table(
  columns: (20.62%, 20.62%, 19.59%, 19.59%, 19.59%),
  align: (left,left,left,left,left,),
  table.header([Final estimator], [Fit time (s)], [Score time (s)], [Train score], [Valid score],),
  table.hline(),
  [LogisticRegression], [19.521 (+/- 1.332)], [0.107 (+/- 0.016)], [0.994 (+/- 0.003)], [0.617 (+/- 0.057)],
  [SVC], [19.582 (+/- 1.081)], [0.120 (+/- 0.020)], [0.983 (+/- 0.010)], [0.614 (+/- 0.022)],
  [XGBClassifier], [20.050 (+/- 1.178)], [0.114 (+/- 0.025)], [0.933 (+/- 0.028)], [0.557 (+/- 0.043)],
)
], caption: figure.caption(
position: top, 
[
Cross-validation results for the 2-model ensemble approach
]), 
kind: "quarto-float-tbl", 
supplement: "Table", 
)
<tbl-stacking-results>


We further separated the header and content features into text and non-text features, training four separate `XGBClassifier` models accordingly. `SVC` was used as the final estimator to aggregate their predictions. #ref(<tbl-architecture-results>, supplement: [Table]) presents the performance comparison across the single-, dual-, and quad-model architectures.

#figure([
#table(
  columns: (23%, 20%, 19%, 19%, 19%),
  align: (left,left,left,left,left,),
  table.header([Architecture], [Fit time (s)], [Score time (s)], [Train score], [Valid score],),
  table.hline(),
  [1 XGBClassifier], [4.075 (+/- 0.307)], [0.098 (+/- 0.017)], [0.998 (+/- 0.001)], [0.605 (+/- 0.071)],
  [2 XGBClassifier + SVC], [19.445 (+/- 1.078)], [0.122 (+/- 0.022)], [0.983 (+/- 0.010)], [0.614 (+/- 0.022)],
  [4 XGBClassifier + SVC], [18.362 (+/- 1.224)], [0.119 (+/- 0.015)], [0.987 (+/- 0.005)], [0.647 (+/- 0.027)],
)
], caption: figure.caption(
position: top, 
[
Cross-validation results for different ensemble architectures
]), 
kind: "quarto-float-tbl", 
supplement: "Table", 
)
<tbl-architecture-results>


=== Approach 2: BERT language models
<approach-2-bert-language-models-1>
tbl-bert-classifiers outlines the results achieved by the BERT language models for classification.

fig-bert-topics visualises the count of topics obtained for a corpus of 100 email bodies.

=== Approach 3: LLMs
<approach-3-llms-1>
The LLM produced accurate classifications and high-quality responses that closely mirrored human reasoning when evaluating emails. However, there are multiple occasions where the quality of responses were poor, where the model showed signs of hallicunations and leakage of training data.

Sample responses can be viewed in #ref(<sec-appendix-3>, supplement: [Section]).

== Proposed data product: PhishSense-v2
<proposed-data-product-phishsense-v2-1>
PhishSense-v2 contains an classifier with a stacked architecture implemented using `StackingClassifier`, comprising four `XGBClassifier` estimators whose predictions are aggregated by a final SVC to produce the final classification. The model takes as input the engineered feature set (split into header and content features), along with separate CountVectorizer representations of the subject line and cleaned plain text.

#ref(<tbl-phishsense-results>, supplement: [Table]) outlines the performance of PhishSense-v2 with the current production model, PhishSense, for comparison.

#figure([
#table(
  columns: (12.71%, 9.32%, 11.02%, 8.47%, 10.17%, 22.88%, 25.42%),
  align: (left,left,right,right,right,right,right,),
  table.header([Model], [Dataset], [Precision], [Recall], [F1-score], [False Benign Rate / FNR], [False Malicious Rate / FPR],),
  table.hline(),
  [PhishSense-v2], [train], [0.887], [0.889], [0.888], [0.046], [0.111],
  [PhishSense-v2], [test], [0.757], [0.748], [0.753], [0.097], [0.252],
  [PhishSense], [test], [0.648], [0.498], [0.563], [0.11], [0.502],
)
], caption: figure.caption(
position: top, 
[
Results of PhishSense and PhishSense-v2
]), 
kind: "quarto-float-tbl", 
supplement: "Table", 
)
<tbl-phishsense-results>


#ref(<fig-conf-mat>, supplement: [Figure]) shows the confusion matrices for PhishSense-v2 and PhishSense evaluated on the test set.

#quarto_super(
kind: 
"quarto-float-fig"
, 
caption: 
[
Confusion matrices obtained from the evaluation on the test set
]
, 
label: 
<fig-conf-mat>
, 
position: 
bottom
, 
supplement: 
"Figure"
, 
subrefnumbering: 
"1a"
, 
subcapnumbering: 
"(a)"
, 
[
#grid(columns: 2, gutter: 2em,
  [
#block[
#figure([
#box(image("../../results/final_results/cm_phishsense-v2_test.png"))
], caption: figure.caption(
position: bottom, 
[
PhishSense-v2
]), 
kind: "quarto-float-fig", 
supplement: "Figure", 
)
<fig-conf-mat-phishsense-v2>


]
],
  [
#block[
#figure([
#box(image("../../results/final_results/cm_phishsense_test.png"))
], caption: figure.caption(
position: bottom, 
[
PhishSense
]), 
kind: "quarto-float-fig", 
supplement: "Figure", 
)
<fig-conf-mat-phishsense>


]
],
)
]
)
= Discussion
<discussion>
== Proposed data science techniques
<proposed-data-science-techniques-2>
=== Approach 1: Classical ML models
<approach-1-classical-ml-models-2>
As shown in #ref(<tbl-classical-results>, supplement: [Table]), `XGBClassifier` outperformed most other models, which can be attirbuted to its ability to capture non-linear relationships between the features and target. However, an inspection of its feature importances revealed a heavy reliance on `CountVectorizer` features, shown in #ref(<fig-xgb-importances>, supplement: [Figure]).

#figure([
#box(image("../../results/base_classifier_selection/xgb_feature_importances.png", height: 2.08333in))
], caption: figure.caption(
position: bottom, 
[
Feature importances for single `XGBClassifier`
]), 
kind: "quarto-float-fig", 
supplement: "Figure", 
)
<fig-xgb-importances>


To address this, we adopted an ensemble approach using a `StackingClassifier` that combines two `XGBClassifier` base estimators -- one trained on email headers and the other on content -- whose predictions are aggregated by a final estimator. The goal was to allow different parts of the email to be independently analysed and improve model generalization. However, #ref(<fig-header-importances>, supplement: [Figure]) and #ref(<fig-body-importances>, supplement: [Figure]) show the base estimators remained overly dependent on CountVectorizer features.

#quarto_super(
kind: 
"quarto-float-fig"
, 
caption: 
[
Feature importances for dual `XGBClassifier`
]
, 
label: 
<fig-dual-xgb-importances>
, 
position: 
bottom
, 
supplement: 
"Figure"
, 
subrefnumbering: 
"1a"
, 
subcapnumbering: 
"(a)"
, 
[
#grid(columns: 2, gutter: 2em,
  [
#block[
#figure([
#box(image("../../results/stacking_final_estimator/header_feat_importances.png", height: 2.08333in))
], caption: figure.caption(
position: bottom, 
[
Header features
]), 
kind: "quarto-float-fig", 
supplement: "Figure", 
)
<fig-header-importances>


]
],
  [
#block[
#figure([
#box(image("../../results/stacking_final_estimator/body_feat_importances.png", height: 2.08333in))
], caption: figure.caption(
position: bottom, 
[
Body features
]), 
kind: "quarto-float-fig", 
supplement: "Figure", 
)
<fig-body-importances>


]
],
)
]
)
To mitigate this, we expanded the ensemble to include four `XGBClassifier` models: each trained on either the header or body and using either engineered features or `CountVectorizer` features. This design aimed to reduce overfitting to the `CountVectorizer` features.

=== Approach 2: BERT language models
<approach-2-bert-language-models-2>
==== Text classification
<text-classification>
Both BERT classifiers struggled with classifying benign emails within the sampled test data, with over half of the malicious emails were misclassified as `benign` by both classifiers.

This poor performance can be attributed to differences in the training data and our dataset. Specifically, the classifiers used were trained with datasets that include legitimate corporate communication emails as the `benign` class. In contrast, both classes in our dataset consists suspicious emails that made it to users' inboxes and reported, resulting in a narrower distinction between benign and malicious examples.

==== Feature engineering
<feature-engineering-1>
The high variance of labels generated by the topic modelling BERT model introduced significant noise, which reduces the reliability of using topic modeling as a feature. Additionally, applying the model to the full training dataset (17,937 emails) would be time-intensive, both in computation and in validating topic accuracy. As such, we decided not to include topic modeling as a feature in our classification task.

==== Sentiment analysis
<sentiment-analysis>
The model labelled most emails as `neutral`, with most empty emails being labelled as `positive`. As such, we concluded that the sentiment analysis BERT model did not provide meaningful signals for our email classification task and decided against incorporating it in our data product.

=== Approach 3: LLMs
<approach-3-llms-2>
A key strength of LLMs is the interpretability of their outputs. Each classification includes a natural language rationale, enhancing transparency and trust. LLMs also require no task-specific training or manual feature engineering and can natively process multilingual text, making them highly adaptable.

Despite these benefits, LLMs face significant challenges in production. The most critical is their high computational cost, where each classification took about 6 minutes, with an additional 2 minutes for summarisation in a resource-unconstrained environment. Each prompt consumed \~45 GB of RAM and fully utilised all 16 CPU cores of the VM. While GPU acceleration could reduce inference time, the resource demands remain prohibitive for production.

Another concern is hallucination, especially under resource constraints involving swap memory that leads to inconsistent outputs. In some cases, signs of training data leakage were observed, with the model repeating phrases or generating nonsensical output until the token limit was reached.

As such, we have decided against incorporating LLMs into PhishSense-v2. Despite that, we have incorporated the insights in feature engineering for classical models.

== Proposed data product: PhishSense-v2
<proposed-data-product-phishsense-v2-2>
While PhishSense-v2 showed measurable improvements over the original model, its performance revealed several important nuances. This includes dataset quality and the influence of probability thresholds on classification outcomes.

=== Dataset quality
<dataset-quality>
==== Similarity between `legitimate` and `spam` emails
<similarity-between-legitimate-and-spam-emails>
Effective model training depends on high-quality data. A key challenge in this project is the similarity between `legitimate` and `spam` emails. Most user-reported emails contained suspicious elements regardless of their label, making it challenging for the model to distinguish between classes. While `spam` emails (#ref(<fig-spam-email>, supplement: [Figure])) appear more suspicious than `legitimate` messages, both are considered as `benign` (refer to #ref(<tbl-target-definitions>, supplement: [Table])). Since the model does relies on the email body, these mixed signals hampers its ability to learn a clear boundary between `benign` and `malicious` bodies.

#figure([
#box(image("../../img/final_report/spam-email.jpg"))
], caption: figure.caption(
position: bottom, 
[
A `spam` email labelled as `benign`
]), 
kind: "quarto-float-fig", 
supplement: "Figure", 
)
<fig-spam-email>


==== Ambiguous relationship between features and label
<ambiguous-relationship-between-features-and-label>
In several instances, the engineered features did not correspond well with the assigned labels. For example, #ref(<fig-ambiguous-emails>, supplement: [Figure]) shows two nearly identical emails (with the same template and 1 minor difference) have similar feature sets, but were labelled differently. These inconsistencies weaken the relationship between features and labels, making it difficult for the model to learn reliable patterns or infer the logic behind label assignments.

#quarto_super(
kind: 
"quarto-float-fig"
, 
caption: 
[
Similar emails in the dataset, but with different labels
]
, 
label: 
<fig-ambiguous-emails>
, 
position: 
bottom
, 
supplement: 
"Figure"
, 
subrefnumbering: 
"1a"
, 
subcapnumbering: 
"(a)"
, 
[
#grid(columns: 2, gutter: 2em,
  [
#block[
#figure([
#box(image("../../img/final_report/ambiguous_benign.jpg"))
], caption: figure.caption(
position: bottom, 
[
Labelled `benign`
]), 
kind: "quarto-float-fig", 
supplement: "Figure", 
)
<fig-ambiguous-emails-benign>


]
],
  [
#block[
#figure([
#box(image("../../img/final_report/ambiguous_malicious.jpg"))
], caption: figure.caption(
position: bottom, 
[
Labelled `malicious`
]), 
kind: "quarto-float-fig", 
supplement: "Figure", 
)
<fig-ambiguous-emails-malicious>


]
],
)
]
)
==== Malicious emails encapsulated within legitimate emails
<malicious-emails-encapsulated-within-legitimate-emails>
Some emails are user-forwarded malicious emails, where a `malicious` email is encapsulated within an otherwise `legitimate` email. In such cases, the email header may appear `benign` while the content is `malicious` which creates conflicting signals.

==== Empty emails
<empty-emails>
Some emails in the dataset had empty bodies, resulting in the model defaulting to predicting `benign`, even when the subject line is suspicious. To address this, we relabelled all empty-content emails as `malicious`, ensuring that future emails with empty content are flagged for manual review.

=== URL features
<url-features>
We initially incorporated URL features such as detecting redirection and verifying the security certificate during prototyping. However, an exploration of the extracted URLs revealed that most of the links lead to web forms that no longer exist or are broken. As web pages are dynamic, it was not possible to obtain the characteristics of the URLs at the time when the email was sent.

=== Probability thresholds
<probability-thresholds>
In PhishSense-v2, the model uses a probability threshold of above \~0.75 to classify emails as `malicious`, requiring high confidence before assigning the `malicious` label. This conservative threshold increases the FNR by misclassifying some `malicious` emails as `benign`. However, lowering the threshold to 0.3 does not significantly affect its performance metrics. To support flexible decision-making, PhishSense-v2 returns the probability values instead.

= Conclusion
<conclusion>
This project introduced a new ML pipeline to streamline the Partner's workflow for reviewing suspicious emails. Our data product, PhishSense-v2, outperformed the existing pipeline in classifying `benign` emails. While further testing and refinement are recommended, the results demonstrate meaningful progress towards automating the email review process and enhancing cybersecurity at UBC.

== Recommendations
<recommendations>
To further improve the model performance, we propose the following recommendations for future work.

=== Audit label assignments in the dataset
<audit-label-assignments-in-the-dataset>
A comprehensive audit of label assignments is essential, particularly in cases where structurally similar emails have been assigned different labels. This process should involve the development of clear and standardised labelling guidelines. Such efforts will help reduce ambiguity and improve abel consistency, thereby strengthening the foundation for model training.

=== Cleaning the dataset
<cleaning-the-dataset>
The dataset should ideally consist only of suspicious emails in its original form, preserving both the header and body content. This helps to minimise noise introduced by forwarded messages, where benign headers accompany malicious content. A cleaner dataset will help the model learn more accurate patterns and reduce wrong predictions.

=== Expand feature set for prediction
<expand-feature-set-for-prediction>
While the team lacks formal training in email security, we believe we have engineered the best possible feature set within the time constraints of this project. Nevertheless, the feature set could be further expanded through collaboration with domain experts. A more comprehensive set of features would provide the model with richer information, potentially improving prediction accuracy.

=== Explore transformer-based models
<explore-transformer-based-models>
Due to limited computational resources, the use of BERT language models was constrained. Future work should revisit these models, as BERT's ability to capture semantic nuances could significantly improve classification accuracy. Fine-tuning on the Partner's dataset may enhance contextual understanding, though it requires a large and well-cleaned dataset.

= Appendix 1
<sec-appendix-1>
#ref(<tbl-feat-extraction>, supplement: [Table]) lists the features that were extracted from the `eml` files.

#figure([
#table(
  columns: (42.86%, 57.14%),
  align: (left,left,),
  table.header([Feature], [Description],),
  table.hline(),
  [`From_email`], [Sender's email],
  [`From_email_domain`], [Sender's email domain],
  [`To_email`], [Recipient's email],
  [`To_email_domain`], [Recipient's email domain],
  [`Subject`], [Email subject line],
  [`Received`], [Routing information of the email],
  [`Authentication-Results`], [Summary results of authentication checks],
  [`Received-SPF`], [SPF check result],
  [`DKIM-Signature`], [DKIM check result],
  [`Content-Language`], [Language declared for the email content],
  [`Reply-To_domain`], [Reply-To email domain],
  [`Content_types`], [Content types present in the email],
  [`text_html`], [HTML content of the email body],
  [`text_plain`], [Plain text content of the email body],
  [`text_clean`], [Cleaned plain text content (with non-UBC email tag and newline characters removed)],
  [`text_hyperlinks`], [List of hyperlinks in the email body],
)
], caption: figure.caption(
position: top, 
[
List of features extracted from `eml` files
]), 
kind: "quarto-float-tbl", 
supplement: "Table", 
)
<tbl-feat-extraction>


#ref(<tbl-feat-eng-header>, supplement: [Table]) and #ref(<tbl-feat-eng-body>, supplement: [Table]) list the features that were engineered from the header and body of the `eml` files respectively.

#figure([
#table(
  columns: (41.67%, 16.67%, 41.67%),
  align: (left,left,left,),
  table.header([Feature], [Data type], [Description],),
  table.hline(),
  [`subject`], [str], [Email subject line.],
  [`url_present_in_subject`], [bool], [If email subject line contains a URL.],
  [`routing_length_before_ubc`], [int], [Number of mail servers the email passed through before reaching a UBC mail server.],
  [`dmarc_authentication_present`], [bool], [If email includes a DMARC result.],
  [`dkim_sender_domains_match`], [bool], [If DKIM signing domain matches the sender's domain.],
  [`to_from_addresses_match`], [bool], [If the recipient's and sender's email addresses are the same.],
  [`sender_email_spf_match`], [bool], [If sender's email domain matches the domain in the SPF results.],
  [`different_reply_domains`], [bool], [If Reply-To domain is different from sender's domain.],
  [`internal_server_transfer_count`], [int \[0, ∞\]], [Number of internal mail servers the email passed through before reaching an external mail server],
  [`name_server_match`], [bool], [True if name servers of sender's domain matches name servers of the mail server that the email originated from.],
  [`dkim_result`], [str], [Result of DKIM check (e.g., pass, fail, none).],
  [`spf_result`], [str], [Result of SPF check (e.g., pass, fail, softfail).],
  [`dmarc_result`], [str], [Result of DMARC check (e.g., pass, fail, none).],
)
], caption: figure.caption(
position: top, 
[
List of features engineered from email headers
]), 
kind: "quarto-float-tbl", 
supplement: "Table", 
)
<tbl-feat-eng-header>


#figure([
#table(
  columns: (41.67%, 16.67%, 41.67%),
  align: (left,left,left,),
  table.header([Feature], [Data type], [Description],),
  table.hline(),
  [`word_count`], [int \[0, ∞\]], [Total number of words in text\_clean.],
  [`readable_proportion`], [float \[0,1\]], [Proportion of text\_clean length over text\_html length.],
  [`whitespace_ratio`], [float \[0,1\]], [Ratio of whitespace or newline characters to total words in text\_clean.],
  [`alphabet_proportion`], [float \[0,1\]], [Proportion of alphabetical to total characters in text\_plain.],
  [`grammar_error_rate`], [float \[0,1\]], [Number of grammatical mistakes in text\_plain, normalised by word\_count.],
  [`english_french_proportion`], [float \[0,1\]], [Proportion of text\_plain that is in English or French.],
  [`text_content_count`], [int \[0, ∞\]], [Number of parts of text-based content (e.g., text/plain, text/html) in the email.],
  [`multimedia_content_count`], [int \[0, ∞\]], [Number of parts of multimedia content (e.g., image/JPEG, video/mpeg) in the email.],
  [`others_content_count`], [int \[0, ∞\]], [Number of parts of all other content (e.g., application/PDF) in the email.],
  [`hyperlink_proportion`], [int \[0, ∞\]], [Number of embedded hyperlinks, normalised by word\_count.],
  [`non_ascii_present`], [bool], [If the text contains non-ASCII characters (e.g., emojis, special symbols).],
  [`hidden_text_present`], [bool], [If text\_html contains hidden or visually obscured text (e.g., white text, small font size).],
  [`empty_body`], [bool], [If email has no text in the body, no hyperlinks, and no multimedia or other content types.],
  [`html_parsing_error`], [int {-1, 0, 1}], [Status code indicating HTML parsing (i.e., -1= exception occurred during parsing, 0 = no error, 1 = error found.)],
)
], caption: figure.caption(
position: top, 
[
List of features engineered from email body
]), 
kind: "quarto-float-tbl", 
supplement: "Table", 
)
<tbl-feat-eng-body>


= Appendix 2
<sec-appendix-2>
F1-score is the harmonic mean of precision (fraction of true positives out of all positive predictions) and recall (fraction of true positives out of all positives). It combines both into a single score that balances the trade-off between false positives and false negatives.

$ upright("F1-score") = frac(2 dot.op upright("precision") dot.op upright("recall"), upright("precision") + upright("recall")) $

$ upright("precision") = frac(T P, T P + F P) \, quad upright("recall") = frac(T P, T P + F N) $

where TP = True Positives, TN = True Negatives, FP = False Positives, FN = False Negatives.

The False Negative (benign) Rate measures the fraction of false negatives out of all positives.

$ upright("False Negative Rate") = frac(F N, T P + F N) $ 

= Appendix 3
<sec-appendix-3>
#ref(<lst-llm-eval-good>, supplement: [Listing]) shows the sample of a good response obtained when reverse-engieering a label.

#figure([
#set align(left)
```md
**Analysis of Email Header and Content for Classification**

**1. Three Distinct Reasons Supporting 'Benign' Classification**  
1. **SPF Pass Validation**: The `received-spf` header confirms the sender's IP (162.19.155.126) is permitted by the domain `kamil.tralewski@craftsplex.pl`, reducing phishing risk.  
2. **Plaintext Content**: The `Content-Type` is `text/plain; charset="utf-8"`, with no attachments or embedded links, minimizing malware delivery potential.  
3. **UBC Infrastructure Handling**: Multiple `Received` headers reference UBC servers (`ebc.ca`, `ubc.ca`, `mail-relay.ubc.ca`), suggesting the email passed through institutional email infrastructure, increasing legitimacy.  

**2. Two Strong Reasons Against 'Malicious' Classification**  
1. **No Attachments/Links**: The absence of attachments or hyperlinks in the email body reduces phishing and malware execution likelihood.  
2. **SPF and DKIM Validation**: The SPF record passes, and the use of institutional email relays (e.g., `Mailbox Transport`, `Frontend Transport`) further supports authenticity.  

**3. One Plausible 'Malicious' Reason**  
1. **Encoded Thread-Topic**: The `Thread-Topic` is Base64-encoded (`=?utf-8?U8WCb3dhIGtsdWN6b3dlIGRvIHd5cG96eWNqb25vd2FuaWE=?=`), which could conceal malicious links or commands when decoded. While not definitive, this encoding is atypical for benign project-related emails and warrants scrutiny.  

**Conclusion**:  
The email exhibits strong indicators of legitimacy (SPF pass, institutional relays, plaintext content) but also raises minor red flags (encoded topic). While the benign classification is strongly supported, the encoded topic remains a plausible concern for malicious intent. Further analysis of the email body and contextual information (e.g., domain reputation of `craftsplex.pl`) would enhance confidence in the classification.
```

], caption: figure.caption(
position: top, 
[
Sample LLM response for reverse engineering
]), 
kind: "quarto-float-lst", 
supplement: "Listing", 
)
<lst-llm-eval-good>


#ref(<lst-llm-classify-good>, supplement: [Listing]) shows the sample of a good response obtained for a classification task.

#figure([
#set align(left)
```json
{
    "label": "phishing", 
    "confidence": "high", 
    "reasons": [
        "Unusual sender address format indicating potential spoofing",
        "Request for immediate action ('your silence is dangerous')",
        "Suspicious email content referencing confidential information and urgent tone"
    ]
}
```

], caption: figure.caption(
position: top, 
[
Sample LLM response for classification
]), 
kind: "quarto-float-lst", 
supplement: "Listing", 
)
<lst-llm-classify-good>


#ref(<lst-llm-eval-bad>, supplement: [Listing]) shows the truncated sample of a bad response obtained from an LLM.

#figure([
#set align(left)
```json
{"label": "phishing", "confidence": "high", "reasons": ["urgent subject creating sense of urgency to provoke immediate action, suspicious domains and use of Mailjet infrastructure commonly associated with phishing campaigns, social engineering tactics to extract sensitive information\', " ," ]}A3. For the function $f(x)=\\frac{2 x^{5}+15 x^{4}-7 x^{2}+3 x-15}{x^{2}+2}$, calculate the derivative at the point $x=-1$. $\\frac{d f(x)}{d x}(x=-1)=$\n(A) 9\n(B) 12\n(C) 13\n(D) 16\n(E) 31\n\nProblems for 3rd yearatterpotterson3adingtoncarolynmissbourdieau
```

], caption: figure.caption(
position: top, 
[
Sample of a poor response from the LLM
]), 
kind: "quarto-float-lst", 
supplement: "Listing", 
)
<lst-llm-eval-bad>


= References
<references>




#bibliography("references-final\_report.bib")

