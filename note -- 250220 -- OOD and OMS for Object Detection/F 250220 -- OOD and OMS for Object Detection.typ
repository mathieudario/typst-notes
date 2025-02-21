// Setting page options
#set text(lang: "en")
#set page(
  paper: "a4",
  margin: (x: 3cm, y: 2cm),
  numbering: "1"
)
#set par(
  leading: .7em,
  spacing: 1.5em,
  justify: true
)

#set heading(numbering: "1.A.a")
#show heading: set block(below: 1em)

// Title
#align(center, text(20pt)[*How to define OOD and OMS detection settings for Object Detection Monitoring ?*])
#v(1em)

// Table of Content
#outline(
  indent: auto,
)
#pagebreak()

// Authors

= Reminder of the Image Classification definitions

In this section, we redefine the concepts of OOD and OMS detection, applied to image classification. Those concepts were primarily defined in [1]. 

Let $T$ our image classification task, defined by an oracle function $Omega$ on an operational domain $X$, i.e. $forall x in X$, the ground-truth of $T$ is $Omega(x)$. Let $f$ be our classification model and $m_f$ the monitor used to reject *unsafe* predictions of $f$.

== OMS detection setting

The scope of $f$ is defined as 

$
D_f = { x in X | f(x) = Omega(x) }
$

And an ideal monitor $m_f^*$ should be defined as 

$
forall x in X, m_f^*(x) = cases(
  0 "if" x in D_f \
  1 "else"
)
$

== OOD detection setting




#pagebreak()
= Generalizing the OMS detection setting

The task of object detection (OD) in computer vision can be seen as the combination of both a classification and a regression task, as the model predicts a bounding box around a detected object as well as the class of this object.

The definition of the scope of the model can no longer be simply put as the domain on which the model prediction is equal to the oracle function. Indeed, for a ML regression task, it is not possible to verify the correctness of the prediction by checking an equality, that happens with probability 0. Instead, one can only verify that a prediction is correct, up to a certain point.

For $x in X$, let's define $N(x)$ the number of predicted objects by the model in this sample. With that, we can define the oracle function as 

$
  forall x in X, Omega(x) = {(c_i,b_i), i in {1, ..., N(x)}}
$

with $c_i$ the predicted class of the object and $b_i$ the predicted localization (bounding box).

Put in simple words, we could say that a prediction $f(x)$ of the model is correct enough (*safe*) if it does not differ too much from the ground-truth (GT) $Omega(x)$. To measure how much the prediction is off the GT, we can introduce an evaluation function (score) $s$ and a threshold $tau$ such that:

$
  forall x in X, f "is correct iff" s(x,f) >= tau
$

The definition of a perfect monitor, which goal is to detect the unsafe prediction, i.e. the ones that are too far from the GT, is:

$
forall x in X, m_f^*(x) 
&= cases(
  0 "if" s(x,f) >= tau \
  1 "else"
)\
&= cases(
  0 "if" x in D_f (tau) \
  1 "else"
)
$
with $D_f (tau) = {x in X | s(x,f) >= tau}$

To define the scoring function $s$, we need to quantify the errors of the model, by validating or not each prediction. To do this, we use the *intersection-over-union* (IoU) and the classification coefficients from confusion matrices (True Positive *TP*, False Positive *FP*, False Negative *FN*). 

Let's fix a sample $x in X$. For each prediction $p$ made by the model on $x$ ($p = (c^p,b^p)$) and each GT label $l$ that is present in $x$ ($l = (c^l,b^l)$), we note $"IoU"(p,l) = "IoU"(b^p,b^l)$ as a simplified notation. We define two thresholds on the IoU, the _background threshold_ $t_b$ and the _foreground threshold_ $t_f$. The evaluation of the prediction $f(x)$ can be broken down into sequential steps.

1. Label assignment

In this step, we try to match each GT label to at most one prediction (detected object). To qualify a positive match, the predicted class must be the same than the GT class, the IoU between the prediction and GT boxes must be greater than $t_f$ (set to $0.5$ by default). In case of multiple predictions being eligible, only the one with greater IoU is selected, the remaining being left unmatched.

In other words, 

$
  forall p, forall l, p "is matched with" l "iff" cases(
    c^p = c^l \
    "IoU"(p,l) >= t_f \
    "IoU"(p,l) = max_(p' | c^(p') = c^l) "IoU"(p',l)
  )
$

2. Error classification 

In this step, we try and determine the number of *TP*, *FP* and *FN*, in order to derive some meaningful metrics, such as the precision (*P*), the recall (*R*) or the $F_1$-score. 

Moreover, let $N_p$ the number of detections made by the model on sample $x$, $N_"GT"$ the number of GT objects to detect in $x$. Let $C$ the number of classes the GT objects can be, and for $i in [1, ..., C]$, let $n_p^i$ the number of detections of class $i$ and $n_"GT"^i$ the number of GT labels of class $i$. 

Each prediction that was matched to a GT label is considered as *TP*.

Each prediction that was left unmatched is considered a *FP*. The classification of the FP can go further. Using the error types classification by [2], we can detail as follow:

- [CLS_ERR] Classification Error : 
$
["IoU"_max >= t_f, "wrong class"]
$
- [LOC_ERR] Localization Error : 
$
  ["IoU"_max in [t_b,t_f], "correct class"]
$
- [LOC_CLS] Localization & Classification Error : 
$
  ["IoU"_max in [t_b,t_f], "wrong class"]
$
- [DUP_ERR] Duplicate Error :
$
  ["IoU"_max >= t_f, "correct class BUT higher scoring pred already matched"]
$
- [BKG_ERR] Background Error :
$
  ["IoU"_max <= t_b, forall "GT label"]
$
- [MIS_ERR] Missed GT Error :
$
  "All the GT label undetected (not matched) and not treated by one error type above."
$

We can therefore define TP as the number of matched predictions, FP as the number of left-unmatched predictions, and FN as the number of missed GT labels (MIS_ERR). For a given object class $i in [1,...,C]$, we also define $"TP"_i$ as the matched detections of class $i$.

We can then define

$
  forall i in [1,...,C], P_i = "TP"_i / n_p^i
$
$
  forall i in [1,...,C], R_i = "TP"_i / n_"GT"^i
$

*Warning*: The above equation are ill-defined. When there is no GT label of class $i$, the definition of $R_i$ does not stand. As well, when there is no detection of class $i$, the definition of $P_i$ does not stand.

To get a metric that combines both the precision and the recall, giving a more complete appreciation of '_how good a prediction of the model is_', we can use the $F_1$-score as the scoring function. To account for the ill-defined metrics above, we can set:

$
  forall i in [1,...,C], S^i (x,f) = cases(
    1 &"if" n_p^i = 0 "and" n_"GT"^i = 0 \
    0 &"if" n_p^i = 0 \
    0 &"if" n_"GT"^i = 0 \
    f^i &"else"
  )
$

with $f^i = 2 dot (P^i dot R^i)/(P^i+R^i)$ the $F_1$ score function

Finally, to get an overall metric on the global prediction (over all classes), we can get mean-score function define below. This mean-score can then be used to define of the model scope $D_f$.

$
  "mS"(x,f) = 1/C sum_(i=1)^C S^i (x,f)
$

