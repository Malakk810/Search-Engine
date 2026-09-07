document.addEventListener("DOMContentLoaded", function () {
    var prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    function animateBars() {
        if (prefersReducedMotion) {
            return;
        }
        var bars = document.querySelectorAll(".bar-fill");
        bars.forEach(function (bar) {
            var target = bar.style.getPropertyValue("--bar-target");
            bar.style.width = "0%";
            requestAnimationFrame(function () {
                requestAnimationFrame(function () {
                    bar.style.width = target;
                });
            });
        });
    }

    function setupExcerptToggles() {
        var excerpts = document.querySelectorAll(".excerpt");
        excerpts.forEach(function (excerpt) {
            var toggle = excerpt.nextElementSibling;
            if (!toggle || !toggle.classList.contains("excerpt-toggle")) {
                return;
            }

            // only show the toggle if the text is actually being clamped
            if (excerpt.scrollHeight <= excerpt.clientHeight + 1) {
                toggle.style.display = "none";
                return;
            }

            toggle.addEventListener("click", function () {
                var expanded = excerpt.classList.toggle("is-expanded");
                toggle.textContent = expanded ? "Show less" : "Read more";
            });
        });
    }

    function setupAutocomplete() {
        var input = document.getElementById("query");
        var ghostTyped = document.querySelector(".ghost-typed");
        var ghostSuggestion = document.querySelector(".ghost-suggestion");
        if (!input || !ghostTyped || !ghostSuggestion) {
            return;
        }

        var debounceTimer = null;
        var currentSuggestion = "";

        function updateGhostTyped() {
            ghostTyped.textContent = input.value;
        }

        function clearSuggestion() {
            currentSuggestion = "";
            ghostSuggestion.textContent = "";
        }

        input.addEventListener("input", function () {
            updateGhostTyped();
            clearSuggestion();
            clearTimeout(debounceTimer);

            var value = input.value;
            if (!value) {
                return;
            }

            debounceTimer = setTimeout(function () {
                fetch("/autocomplete?q=" + encodeURIComponent(value))
                    .then(function (response) {
                        return response.json();
                    })
                    .then(function (data) {
                        if (input.value !== value) {
                            return;
                        }
                        currentSuggestion = data.completion || "";
                        ghostSuggestion.textContent = currentSuggestion;
                    })
                    .catch(function () {
                    });
            }, 250);
        });

        input.addEventListener("keydown", function (event) {
            var atEnd = input.selectionStart === input.value.length;
            if ((event.key === "Tab" || event.key === "ArrowRight") && currentSuggestion && atEnd) {
                event.preventDefault();
                input.value += currentSuggestion;
                updateGhostTyped();
                clearSuggestion();
            }
        });

        updateGhostTyped();
    }

    animateBars();
    setupAutocomplete();
    if (document.fonts && document.fonts.ready) {
        document.fonts.ready.then(setupExcerptToggles);
    } else {
        setupExcerptToggles();
    }
});