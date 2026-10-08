{{ fullname | escape | underline}}

.. automodule:: {{ fullname }}

   {% block attributes %}
   {%- if attributes %}
   .. rubric:: {{ _('Module Attributes') }}

   .. autosummary::
      :signatures: none
      :toctree: generated/
   {% for item in attributes %}
   {%- if item not in functions and item not in classes and item not in exceptions %}
      {{ item }}
   {%- endif %}
   {%- endfor %}
   {% endif %}
   {%- endblock %}

   {%- block functions %}
   {%- if functions %}
   .. rubric:: {{ _('Functions') }}

   .. autosummary::
      :signatures: short
      :toctree: generated/
   {% for item in functions %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {%- endblock %}

   {%- block classes %}
   {%- if classes %}
   .. rubric:: {{ _('Classes') }}

   .. autosummary::
      :signatures: none
      :toctree: generated/
   {% for item in classes %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {%- endblock %}

   {%- block exceptions %}
   {%- if exceptions %}
   .. rubric:: {{ _('Exceptions') }}

   .. autosummary::
      :signatures: none
      :toctree: generated/
   {% for item in exceptions %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {%- endblock %}