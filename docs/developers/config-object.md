# an Armory Configuration object

As of armory 0.14.x application configuration is built up from various sources
and bits of it are sprinkled through the armory application. This results in
application state distributed hodge-podge through the app. Because the app
configuration is built piece-wise by multiple modules, proper unit test rigging
is difficult or impossible without hoisting the whole system up first.

This note outlines a configuration object to contain and render all application
configuration state.
