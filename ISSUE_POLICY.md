# Issue Policy

The Oracle Guardian AI Issue Policy outlines the categories of Oracle Guardian AI GitHub issues and discusses the guidelines and processes associated with each type of issue.

Before filing an issue, make sure to [search for related issues](https://github.com/oracle/guardian-ai/issues) and check if they address the same problem you're encountering.

## Issue Categories

Our policy states that GitHub issues fall into the following categories:

1. Feature Requests
2. Bug Reports
3. Documentation Fixes
4. Installation Issues

Each category has its own GitHub issue template. Please refrain from deleting the issue template unless you are certain that your issue does not fit within its scope.

### Feature Requests

#### Guidelines

To increase the likelihood of having a feature request accepted, please ensure that:

- The request has a minimal scope (note that it's easier to add additional functionality later than to remove functionality).
- The request has a significant impact on users and provides value that justifies the maintenance efforts required to support the feature in the future.

#### Lifecycle

Feature requests typically go through the following stages:

1. Submit a feature request GitHub Issue, providing a brief overview of the proposal and its motivation. If possible, include an implementation overview as well.
2. The issue will be triaged to determine if more information is needed from the author, assign a priority, and route the request to the appropriate committers.
3. Discuss the feature request with a committer who will provide input on the implementation overview or request a more detailed design if necessary.
4. Once there is agreement on the feature request and its implementation, an implementation owner will be assigned.
5. The implementation owner will start developing the feature and ultimately submit associated pull requests to the Oracle Guardian AI Repository.

### Bug Reports

#### Guidelines

To ensure that maintainers can effectively assist with any reported bugs, please follow these guidelines:

- Fill out the bug report template completely, providing appropriate levels of detail, especially in the "Code to reproduce issue" section.
- Verify that the bug you are reporting meets one of the following criteria:
  - It is a regression where a recent release of Oracle Guardian AI no longer supports an operation that was supported in an earlier release.
  - A documented feature or functionality does not work as intended when executing a provided example from the documentation.
  - Any raised exception is directly from Oracle Guardian AI and not the result of an underlying package's exception.
- Make an effort to diagnose and troubleshoot the issue before filing the report.
- Ensure that the environment in which you encountered the bug is supported as defined in the documentation.
- Validate that Oracle Guardian AIports the functionality you are experiencing issues with. Remember that the absence of a feature does not constitute a bug.
- Read the documentation for the feature related to the issue you are reporting. If you are certain that you are following the documented guidelines, please file a bug report.

#### Lifecycle

Bug reports typically go through the following stages:

1. Submit a bug report GitHub Issue, providing a high-level description of the bug and all the necessary information to reproduce it.
2. The bug report will be triaged to determine if more information is required from the author, assign a priority, and route the issue to the appropriate committers.
3. An Oracle Guardian AI committer will reproduce the bug and provide feedback on how to implement a fix.
4. Once an approach has been agreed upon, an owner for the fix will be assigned. For severe bugs, Oracle Guardian AI committers may choose to take ownership to ensure a timely resolution.
5. The fix owner will start implementing the solution and ultimately submit associated pull requests.

### Documentation Fixes

#### Lifecycle

Documentation issues typically go through the following stages:

1. Submit a documentation GitHub Issue, describing the issue and indicating its location(s) in the Oracle Guardian AI documentation.
2. The issue will be triaged to determine if more information is needed from the author, assign a priority, and route the request to the appropriate committers.
3. An Oracle Guardian AI committer will confirm the documentation issue and provide feedback on how to implement a fix.
4. Once an approach has been agreed upon, an owner for the fix will be assigned. For severe documentation issues, Oracle Guardian AI committers may choose to take ownership to ensure a timely resolution.
5. The fix owner will start implementing the solution and ultimately submit associated pull requests.

### Installation Issues

#### Lifecycle

Installation issues typically go through the following stages:

1. Submit an installation GitHub Issue, describing the issue and indicating the platforms it affects.
2. The issue will be triaged to determine if more information is needed from the author, assign a priority, and route the issue to the appropriate committers.
3. An Oracle Guardian AI committer will confirm the installation issue and provide feedback on how to implement a fix.
4. Once an approach has been agreed upon, an owner for the fix will be assigned. For severe installation issues, Oracle Guardian AI committers may choose to take ownership to ensure a timely resolution.
5. The fix owner will start implementing the solution and ultimately submit associated pull requests.
