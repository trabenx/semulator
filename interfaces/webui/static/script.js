document.addEventListener('DOMContentLoaded', () => {
    const configFormContainer = document.getElementById('config-form-container');
    const configDataElement = document.getElementById('config-data');
    const generateForm = document.getElementById('generate-form');
    const previewButton = document.getElementById('preview-button');
    const previewArea = document.getElementById('preview-area');
    const previewError = document.getElementById('preview-error');
    const exportButton = document.getElementById('export-config-button');
    const importInput = document.getElementById('import-config-input');
    const startTaskStatus = document.getElementById('start-task-status');

    let baseConfig = {};

    // --- Helper: Create Accordion ---
    function createAccordionItem(title, parentElement) {
        const itemDiv = document.createElement('div');
        itemDiv.className = 'accordion-item';

        const headerButton = document.createElement('button');
        headerButton.type = 'button'; // Important to prevent form submission
        headerButton.className = 'accordion-header';
        headerButton.textContent = title;

        const contentDiv = document.createElement('div');
        contentDiv.className = 'accordion-content';
        contentDiv.style.display = 'none'; // Start collapsed

        itemDiv.appendChild(headerButton);
        itemDiv.appendChild(contentDiv);
        parentElement.appendChild(itemDiv);

        // Add click listener to toggle
        headerButton.addEventListener('click', () => {
            const isVisible = contentDiv.style.display === 'block';
            contentDiv.style.display = isVisible ? 'none' : 'block';
            headerButton.classList.toggle('active', !isVisible);
        });

        return contentDiv; // Return the content div to append form fields into
    }

     // --- Helper: Format Label ---
    function formatLabel(key) {
        // Replace _range, _choices, _probability suffixes for display
        key = key.replace(/_(range|choices|probability)$/, '');
        // Replace other underscores, capitalize
        return key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
    }

    // --- Keep createFormRow (updated label formatting) ---
    function createFormRow(labelText, parentElement) {
        const row = document.createElement('div');
        row.className = 'form-row';
        const label = document.createElement('label');
        label.textContent = labelText; // Already formatted
        // label.htmlFor = ???; // Hard to link dynamically here
        row.appendChild(label);
        parentElement.appendChild(row);
        return row; // Return the div for appending inputs
    }

    function setNestedValue(obj, path, value) {
        const keys = path.split('.');
        let current = obj;
        for (let i = 0; i < keys.length - 1; i++) {
            const key = keys[i];
            // Handle potential list indices represented as keys
            const nextKey = keys[i + 1];
            const nextIsIndex = !isNaN(parseInt(nextKey)); // Check if next key is a number

            if (nextIsIndex) {
                // Ensure current level is an array
                if (!Array.isArray(current[key])) {
                    current[key] = [];
                }
            } else {
                 // Ensure current level is an object
                 if (typeof current[key] === 'undefined' || current[key] === null) {
                     current[key] = {};
                 } else if (Array.isArray(current[key])) {
                      // Trying to set an object property on an array index? Ambiguous.
                      // For now, assume we create objects at numerical indices if needed
                      if(typeof current[key][parseInt(key)] === 'undefined') {
                          current[key][parseInt(key)] = {}; // This case needs careful thought based on structure
                      }
                 } else if(typeof current[key] !== 'object') {
                      current[key] = {}; // Overwrite if not object/array
                 }
            }
            // Handle array index access
            if(nextIsIndex){
                const index = parseInt(nextKey);
                 if(current[key].length <= index){
                      // Pad array with nulls or empty objects if index is out of bounds
                      for(let j=current[key].length; j<=index; j++){
                           // Look ahead: if the key AFTER the index is a property name, create object
                           if(keys.length > i+2 && isNaN(parseInt(keys[i+2]))){
                               current[key].push({});
                           } else {
                               current[key].push(null); // Otherwise assume primitive/list later
                           }
                      }
                 }
                 current = current[key][index];
                 i++; // Skip the index key in the next iteration
            } else {
                 current = current[key];
            }
        }
        // Set the final value
        current[keys[keys.length - 1]] = value;
    }

    // --- 1. Build Form Dynamically ---
    function buildForm(configValue, parentElement, prefix = '') {
        // console.log(`buildForm called for prefix: '${prefix}', type: ${typeof configValue}`); // Debug

        if (Array.isArray(configValue)) {
            // --- Handle Arrays (Layers, Artifacts, etc.) ---
            if (configValue.length > 0 && typeof configValue[0] === 'object' && configValue[0] !== null) {
                // Create accordion items for list of objects
                configValue.forEach((item, index) => {
                    const itemPrefix = `${prefix}.${index}`;
                    const itemName = item.layer_id || item.name || `Item ${index + 1}`; // Use index+1 for display
                    const accordionContent = createAccordionItem(itemName, parentElement);
                    buildForm(item, accordionContent, itemPrefix); // Recurse into the item object
                });
            } // Ignore primitive arrays unless handled by specific keys (_range/_choices)

        } else if (typeof configValue === 'object' && configValue !== null) {
            // --- Handle Dictionaries (Nested Sections) ---
            const keys = Object.keys(configValue).sort();
            // Determine the container for this object's keys
            // Only create accordions for top-level sections or specific nested ones like categories
            const isTopLevel = prefix === '';
            const isCategoryListContainer = prefix.endsWith('.categories'); // e.g., artifact_raffle.categories
            let currentContainer = parentElement;

            if (isTopLevel && Object.keys(configValue).length > 0) {
                 // Don't create an accordion for the absolute root, process keys directly
                 currentContainer = parentElement;
            }
             // --- NO single "General Settings" accordion needed if isTopLevel handles direct keys ---
             // else if (!prefix.endsWith('.params') && prefix !== '' && !isCategoryListContainer) {
             //      // Create accordion for nested dictionary sections
             //      const sectionTitle = prefix.split('.').pop();
             //      currentContainer = createAccordionItem(formatLabel(sectionTitle), parentElement);
             // }


            keys.forEach(key => {
                // Skip internal keys
                if (key.startsWith('_') || key === 'selected_layers') return;

                const value = configValue[key];
                const currentPrefix = prefix ? `${prefix}.${key}` : key;
                const labelText = formatLabel(key); // Format label early

                // --- Process based on key suffix or value type ---

                // Special handling for lists within categories (e.g., shape, geometric)
                if (isCategoryListContainer && Array.isArray(value)) {
                     const categoryFieldset = document.createElement('fieldset');
                     const categoryLegend = document.createElement('legend');
                     categoryLegend.textContent = labelText; // e.g., "Shape", "Geometric"
                     categoryFieldset.appendChild(categoryLegend);
                     // Recursively call buildForm for the list itself
                     buildForm(value, categoryFieldset, currentPrefix);
                     currentContainer.appendChild(categoryFieldset); // Add to parent (e.g., 'artifact_raffle' accordion)
                     return; // Handled this key, move to next key in categories
                }

                // Skip recursing into 'params' dictionary itself, process its keys directly
                if (key === 'params' && typeof value === 'object' && value !== null) {
                     buildForm(value, currentContainer, prefix); // Use PARENT's prefix & container
                     return;
                 }

                // Handle specific key types (_range, _choices, booleans)
                if (key.endsWith('_range') && Array.isArray(value) && value.length === 2) {
                    // Create Slider
                    const baseKey = key.replace('_range', '');
                    const sliderId = prefix ? `${prefix}.${baseKey}` : baseKey;
                    const row = createFormRow(formatLabel(baseKey), currentContainer); // Use baseKey label
                    const slider = document.createElement('input');
					slider.type = 'range';
					slider.id = sliderId;
					slider.name = sliderId;
					
                    const minVal = Number(value[0]); // Ensure numbers
                    const maxVal = Number(value[1]);
                    const isFloat = !Number.isInteger(minVal) || !Number.isInteger(maxVal);
                    const range = maxVal - minVal;
					
                    slider.min = minVal;
					slider.max = maxVal;
                    // Ensure step is valid and not zero
                    let step = isFloat ? (range / 100) : 1;
                    if (step <= 0) { step = isFloat ? 0.01 : 1; } // Fallback step
                    slider.step = step.toPrecision(2);

                    // Set default value: Check if baseKey exists (meaning override), else use midpoint
                    const defaultValue = configValue[baseKey] !== undefined ? configValue[baseKey] : (minVal + range / 2);
                    slider.value = defaultValue;
                    // console.log(`Slider ${sliderId}: min=${minVal}, max=${maxVal}, step=${step}, default=${defaultValue}`); // Debug

                    const valueDisplay = document.createElement('span');
                    valueDisplay.style.marginLeft = '10px';
                    valueDisplay.style.minWidth = '40px'; // Reserve space
                    valueDisplay.style.display = 'inline-block'; // Allow width
                    valueDisplay.textContent = parseFloat(slider.value).toFixed(isFloat ? 2 : 0);

                    slider.oninput = () => {
                        valueDisplay.textContent = parseFloat(slider.value).toFixed(isFloat ? 2 : 0);
                    };
                    row.appendChild(slider);
                    row.appendChild(valueDisplay);
					return;
                } else if (key.endsWith('_choices') && Array.isArray(value)) {
                    // Create Dropdown
                     const baseKey = key.replace('_choices', '');
                     const selectId = `${prefix}.${baseKey}`;
                     const row = createFormRow(formatLabel(baseKey), currentContainer);
                     const select = document.createElement('select'); select.id = selectId; select.name = selectId;
                     value.forEach(choice => { const option = document.createElement('option'); option.value = choice; option.textContent = choice; if (configValue[baseKey] === choice) { option.selected = true; } select.appendChild(option); });
                     row.appendChild(select);

                } else if ((key === 'enabled' || typeof value === 'boolean') && key !== 'verbose') {
                    // Create Checkbox
                     const checkboxId = currentPrefix;
                     const row = createFormRow(labelText, currentContainer);
                     const checkbox = document.createElement('input'); checkbox.type = 'checkbox'; checkbox.id = checkboxId; checkbox.name = currentPrefix; checkbox.checked = Boolean(value); // Ensure boolean
                     row.appendChild(checkbox);

                } else if (typeof value === 'object' && value !== null) {
                     // --- RECURSE FOR OTHER NESTED OBJECTS ---
                     // Create an accordion for this nested section
                     const nestedAccordionContent = createAccordionItem(labelText, currentContainer);
                     buildForm(value, nestedAccordionContent, currentPrefix); // Recurse
                     // ---

                } else if (key !== 'layer_id' && key !== 'name' && key !== 'shape' && key !== 'type' && !key.endsWith('_choices') && !key.endsWith('_range') && !key.endsWith('_probability')) {
                    // Create Default Input (Text/Number)
                    // Handle probability separately if needed (e.g., as range 0-1)
                    const inputId = currentPrefix;
                    const row = createFormRow(labelText, currentContainer);
                    const input = document.createElement('input');
                    if (typeof value === 'number') { input.type = 'number'; input.step = 'any'; }
                    else { input.type = 'text'; }
                    input.id = inputId; input.name = currentPrefix;
                    input.value = value;
                    row.appendChild(input);
                }
                 // Note: probability keys are implicitly handled by the default text/number input above
                 // unless they had a _range or _choices suffix (which were handled).
            }); // End keys.forEach

        }
        // else primitive value - skip (should have been handled by parent call)
    }

    // --- 2. Update Form from Imported JSON ---
    function updateForm(importedData, prefix = '') {
        // console.log(`Update form: prefix='${prefix}'`, importedData); // Debug
        Object.keys(importedData).forEach(key => {
            const value = importedData[key];
            const currentPrefix = prefix ? `${prefix}.${key}` : key;

             if (key.startsWith('_') || key === 'selected_layers' || key === 'params') {
                 if (key === 'params' && typeof value === 'object' && value !== null){
                      updateForm(value, prefix); // Recurse into params value with parent prefix
                 }
                 return;
             }

            // --- Recurse for nested structures ---
            if (Array.isArray(value)) {
                if (value.length > 0 && typeof value[0] === 'object') {
                     value.forEach((item, index) => {
                         const itemPrefix = `${currentPrefix}.${index}`;
                         if(document.querySelector(`[name^="${itemPrefix}"]`)){ updateForm(item, itemPrefix); }
                         else { console.warn(`UpdateForm: Form elements for list item ${itemPrefix} not found.`); }
                     });
                 }
                 // Skip primitive arrays unless part of range/choice handled below

            } else if (typeof value === 'object' && value !== null) {
                 // Check if form elements for this nested object exist before recursing
                 // This helps avoid errors if the form structure doesn't perfectly match JSON
                 if(document.querySelector(`[name^="${currentPrefix}."]`) || document.querySelector(`[name="${currentPrefix}"]`)){
                    updateForm(value, currentPrefix);
                 } else {
                    // console.warn(`UpdateForm: No form elements found starting with prefix ${currentPrefix}, skipping recursion.`);
                 }

            } else {
                // --- Update individual primitive form elements ---
                const baseKey = key.replace(/_(range|choices|probability)$/, '');
                const elementPath = prefix ? `${prefix}.${baseKey}` : baseKey;

                // Find potential elements (slider uses elementPath, checkbox uses currentPrefix)
                const sliderElement = document.querySelector(`[name="${elementPath}"][type="range"]`);
                const checkboxElement = document.querySelector(`[name="${currentPrefix}"][type="checkbox"]`);
                // Select/Input/Number use elementPath OR currentPrefix depending on buildForm logic
                const otherElement = document.querySelector(`[name="${elementPath}"]:not([type="checkbox"]):not([type="range"])`) || document.querySelector(`[name="${currentPrefix}"]:not([type="checkbox"]):not([type="range"])`);

                // logger.debug(`Trying update: key='${key}', path='${elementPath}', currentPrefix='${currentPrefix}', value='${value}'`); // Debug

                if (sliderElement && (key.endsWith('_range') || typeof value === 'number')) {
                     // logger.debug(`Updating slider ${sliderElement.name} to ${value}`); // Debug
                     // If the source JSON has the _range key, use its midpoint as default if single value not present
                     let sliderValue = value;
                     if(Array.isArray(value) && value.length === 2){
                          sliderValue = (value[0] + value[1]) / 2; // Use midpoint from range
                     }
                     sliderElement.value = sliderValue;
                     const displaySpan = sliderElement.nextElementSibling;
                     if (displaySpan && displaySpan.tagName === 'SPAN') {
                         const isFloat = sliderElement.step && sliderElement.step !== '1';
                         displaySpan.textContent = parseFloat(sliderElement.value).toFixed(isFloat ? 2 : 0);
                     }
                } else if (checkboxElement && typeof value === 'boolean') {
                     // logger.debug(`Updating checkbox ${checkboxElement.name} to ${value}`); // Debug
                     checkboxElement.checked = value;
                } else if (otherElement) {
                     if (otherElement.tagName === 'SELECT') {
                          // logger.debug(`Updating select ${otherElement.name} to ${value}`); // Debug
                          otherElement.value = String(value);
                     } else { // Text or Number input
                          // logger.debug(`Updating input ${otherElement.name} to ${value}`); // Debug
                          otherElement.value = value;
                     }
                }
            }
        });
    }



    // --- 3. Get Form Data as JSON for Export/Preview/Start ---
    function getFormDataAsOverrides(formElement) {
        const formData = new FormData(formElement);
        const overrides = {};
        // NO rangePartials needed anymore - sliders submit single values directly

        formData.forEach((valueStr, key) => {
            // ... (Basic parsing: numeric, boolean, string - same as before) ...
            let parsedValue = valueStr;
            const inputElement = formElement.querySelector(`[name="${key}"]`); // Get element info
            const isCheckbox = inputElement && inputElement.type === 'checkbox';
            const isRangeInput = inputElement && inputElement.type === 'range'; // Slider check

            if (isCheckbox) {
                 parsedValue = (valueStr === 'on');
            } else if (valueStr !== null && valueStr.trim() !== '') { // Ensure not empty before checking number
                if (!isNaN(valueStr)) { // isNaN handles ints and floats
                    parsedValue = Number(valueStr);
                } else if (valueStr.toLowerCase() === 'true') { parsedValue = true; }
                else if (valueStr.toLowerCase() === 'false') { parsedValue = false; }
            } else if (valueStr === '') {
                 return; // Skip empty strings
            }
             // else keep as string if needed, though most should be parsed

            // Use the existing setNestedValue helper
            setNestedValue(overrides, key, parsedValue);
        });
        return overrides;
    }

    // --- 4. Event Handlers ---

    // Preview
    if (previewButton) {
        previewButton.addEventListener('click', async () => {
            previewArea.textContent = 'Generating Preview...';
            previewArea.style.display = 'flex'; // Ensure visible
            previewError.textContent = '';
            previewButton.disabled = true;
            generateForm.querySelector('button[type=submit]').disabled = true;

            //const formData = new FormData(generateForm);
            try {
                // --- Get only the overrides ---
                const formOverrides = getFormDataAsOverrides(generateForm);
                // --- Create the payload ---
                const payload = {
                    base_config: baseConfig, // Send original base config
                    overrides: formOverrides // Send only changed values
                };
                // ---

                const response = await fetch('/preview', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload) // Send combined payload
                });
                const data = await response.json();

                if (response.ok && data.status === 'success') {
                    const img = document.createElement('img');
                    img.src = `data:${data.mime_type};base64,${data.image_data}`;
                    img.style.maxWidth = '100%';
                    img.style.maxHeight = '400px'; // Limit preview size
                    previewArea.innerHTML = ''; // Clear text
                    previewArea.appendChild(img);
                } else {
                    previewArea.textContent = 'Preview Failed';
                    previewError.textContent = data.message || 'Unknown error during preview.';
                }
            } catch (error) {
                previewArea.textContent = 'Preview Failed';
                previewError.textContent = `Network or server error: ${error}`;
            } finally {
                previewButton.disabled = false;
                generateForm.querySelector('button[type=submit]').disabled = false;
            }
        });
    }

    // Start Task (Form Submit)
    if (generateForm) {
        generateForm.addEventListener('submit', async (event) => {
            event.preventDefault(); // Prevent default HTML form submission
            startTaskStatus.textContent = 'Starting task...';
            startTaskStatus.style.color = 'black';
            previewButton.disabled = true;
            generateForm.querySelector('button[type=submit]').disabled = true;


            //const formData = new FormData(generateForm);
            try {
                // --- Get only the overrides ---
                const formOverrides = getFormDataAsOverrides(generateForm);
                // --- Create the payload ---
                const payload = {
                    base_config: baseConfig, // Send original base config
                    overrides: formOverrides // Send only changed values
                };
                // ---

                const response = await fetch('/start_task', {
                    method: 'POST',
                     headers: { 'Content-Type': 'application/json' },
                     body: JSON.stringify(payload) // Send combined payload
                });
                const data = await response.json();

                if (response.ok && data.status === 'success') {
                    startTaskStatus.textContent = `Task ${data.task_id} started successfully! Redirecting to tasks page...`;
                    startTaskStatus.style.color = 'green';
                    // Redirect to tasks page after a short delay
                    setTimeout(() => {
                         window.location.href = '/tasks';
                    }, 1500);
                } else {
                    startTaskStatus.textContent = `Error starting task: ${data.message || 'Unknown error.'}`;
                    startTaskStatus.style.color = 'red';
                    previewButton.disabled = false;
                    generateForm.querySelector('button[type=submit]').disabled = false;
                }
            } catch (error) {
                 startTaskStatus.textContent = `Network or server error: ${error}`;
                 startTaskStatus.style.color = 'red';
                 previewButton.disabled = false;
                 generateForm.querySelector('button[type=submit]').disabled = false;
            }
        });
    }

    // Export Config
    if (exportButton) {
        exportButton.addEventListener('click', () => {
            try {
                 // Get overrides and merge them into a *copy* of baseConfig for export
                 const formOverrides = getFormDataAsOverrides(generateForm);
                 // Need a JS deep merge/update function here, or send to backend for merging?
                 // Simple approach: just export the overrides for now? Or rely on baseConfig?
                 // Let's export the current overrides merged into base config
                 // Need a simple deep merge JS function:
                 function jsDeepUpdate(target, source) {
                    const output = Object.assign({}, target); // Shallow copy initially
                    if (typeof target === 'object' && target !== null && typeof source === 'object' && source !== null) {
                        Object.keys(source).forEach(key => {
                            const targetValue = target[key];
                            const sourceValue = source[key];
                            if (typeof targetValue === 'object' && targetValue !== null && typeof sourceValue === 'object' && sourceValue !== null && !Array.isArray(sourceValue)) {
                                output[key] = jsDeepUpdate(targetValue, sourceValue); // Recurse for objects
                            } else {
                                output[key] = sourceValue; // Overwrite otherwise (handles primitives, arrays)
                            }
                        });
                    }
                    return output;
                 }
                 const currentFullConfig = jsDeepUpdate(baseConfig, formOverrides); // Merge overrides onto base

                 const jsonString = JSON.stringify(currentFullConfig, null, 4);

                 // Create a Blob and download link
                 const blob = new Blob([jsonString], { type: 'application/json' });
                 const url = URL.createObjectURL(blob);
                 const a = document.createElement('a');
                 a.href = url;
                 a.download = 'semgen_config_export.json';
                 document.body.appendChild(a);
                 a.click();
                 document.body.removeChild(a);
                 URL.revokeObjectURL(url);
            } catch (error) {
                 console.error("Error exporting config:", error);
                 alert("Error exporting configuration. Check console.");
            }

        });
    }

    // Import Config
    if (importInput) {
        importInput.addEventListener('change', (event) => {
            const file = event.target.files[0];
            if (!file) return;

            const reader = new FileReader();
            reader.onload = (e) => {
                try {
                    const importedConfig = JSON.parse(e.target.result);
                    // When importing, REPLACE the baseConfig and rebuild the form
                    baseConfig = importedConfig; // Update the stored base config
                    configFormContainer.innerHTML = ''; // Clear existing form
                    buildForm(baseConfig, configFormContainer); // Rebuild form from imported config
                    alert('Configuration imported successfully and form updated!');
                } catch (error) {
                    console.error("Error importing config:", error);
                    alert(`Error importing configuration file: ${error.message}`);
                } finally {
                     // Reset file input so the same file can be selected again if needed
                     importInput.value = '';
                }
            };
            reader.onerror = (e) => {
                 alert(`Error reading file: ${e}`);
                 importInput.value = '';
            };
            reader.readAsText(file);
        });
    }


    // --- Initial Setup ---
    if (configDataElement && configFormContainer) {
        try {
            // Store the initial config loaded from the template
            baseConfig = JSON.parse(configDataElement.textContent);
            configFormContainer.innerHTML = '';
            buildForm(baseConfig, configFormContainer);
        } catch (error) {
            configFormContainer.textContent = 'Error loading/parsing base configuration.';
            console.error("Error parsing base config:", error);
        }
    } else {
         console.error("Required elements for form building not found.");
    }
});
