.. Template for release notes. TODO: fill in the blanks and remove comments.

==============
Release Notes
==============

1.3.0
-----

Release date: March 17, 2025

**New Features and Enhancements:**

* Added new support for LLMs in the fairness module for measuring toxicity bias in LLMs. These metrics measure the disparity in toxic generations -- that is, whether or not your LLM is more toxic when talking about one group of people than another. 

1.2.0
-----

Release date: November 12, 2024

* Upgraded scikit-learn to 1.5.0

1.1.0
-----

Release date: April 22, 2024

**New Features and Enhancements:**

* Enhanced bias mitigation to avoid solutions with levelling down (that is, making outcomes worse for) one or more groups to achieve fairness metric rate parity.

* Added warm starting mechanism to bias mitigation to reduce the time required to find high-quality solution trade-offs.

* Replaced ``AIF360`` rate-based fairness metrics with in-house ones to improve running times.


1.0.1
-----

Release date: December 8, 2023

**Bug Fixes:**

* Fixed a bug in the rate-based fairness metrics that caused them to report incomplete results when using ``reduction=None``.


1.0.0
-----

Release date: Oct 13, 2023

**New Features and Enhancements:**

* Initial repository.
