Andre's libCellML Python Utils Properties
=========================================

Here we define the properties that are used in annotations by these utilities.
Ideally these will be replaced with more standardised identifiers at some point.

variable-to-delay
-----------------

:code:`https://github.com/nickerso/libcellml-python-utils/properties.rst#variable-to-delay`

Used to indicate a variable in a model that should be delayed. Should be used as the `object` in an RDF triple with the variable as the `subject` and the :code:`dcterms:type` `predicate`.
For example:

.. code::

    @prefix dcterms: <http://purl.org/dc/terms/> .
    @prefix aprop: <https://github.com/nickerso/libcellml-python-utils/properties.rst#> .

    <sine_approximations-updated-ids.xml#b4da83> dcterms:type
        aprop:variable-to-delay .

Variables to be delayed should also have a :code:`delay-variable` annotation to define the delay variable
and a :code:`delay-amount-variable`` to define the delay amount.

delay-variable
--------------

:code:`https://github.com/nickerso/libcellml-python-utils/properties.rst#delay-variable`

Used as the `predicate` in an RDF triple to link the variable to be delayed (`subject`) to the delay variable (`object`).
For example:

.. code::

    @prefix dcterms: <http://purl.org/dc/terms/> .
    @prefix aprop: <https://github.com/nickerso/libcellml-python-utils/properties.rst#> .

    <sine_approximations-updated-ids.xml#b4da83> dcterms:type
        aprop:variable-to-delay ;
        aprop:delay-variable <sine_approximations-updated-ids.xml#b4da82> .

delay-amount-variable
--------------

:code:`https://github.com/nickerso/libcellml-python-utils/properties.rst#delay-amount-variable`

Used as the `predicate` in an RDF triple to link the variable to be delayed (`subject`) to the delay amount variable (`object`).
For example:

.. code::

    @prefix dcterms: <http://purl.org/dc/terms/> .
    @prefix aprop: <https://github.com/nickerso/libcellml-python-utils/properties.rst#> .

    <sine_approximations-updated-ids.xml#b4da83> dcterms:type
        aprop:variable-to-delay ;
        aprop:delay-amount-variable <sine_approximations-updated-ids.xml#b4da82> .

independent-variable
--------------

:code:`https://github.com/nickerso/libcellml-python-utils/properties.rst#independent-variable`

Used as the `predicate` in an RDF triple to link the variable to be delayed (`subject`) to the independent variable (`object`).
For example:

.. code::

    @prefix dcterms: <http://purl.org/dc/terms/> .
    @prefix aprop: <https://github.com/nickerso/libcellml-python-utils/properties.rst#> .

    <sine_approximations-updated-ids.xml#b4da83> dcterms:type
        aprop:variable-to-delay ;
        aprop:delay-amount-variable <sine_approximations-updated-ids.xml#b4da82> .
