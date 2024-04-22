.. Template for release notes. TODO: fill in the blanks and remove comments.

==============
Release Notes
==============

1.1.0
-----

Release date: April 22, 2023

**New Features and Enhancements:**

* Enhanced bias mitigation to avoid solutions that levelling down (that is, making outcomes worse for) one or more groups to achieve fairness metric rate parity.

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
