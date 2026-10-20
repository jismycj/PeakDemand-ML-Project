document.addEventListener(
    "DOMContentLoaded",
    function () {

        const url =
            new URL(
                window.location.href
            );

        const prediction =
            url.searchParams.get(
                "prediction"
            );

        if (prediction) {

            const section =
                document.querySelector(
                    ".latest-section"
                );

            if (section) {

                setTimeout(
                    function () {

                        section.scrollIntoView(
                            {
                                behavior: "smooth"
                            }
                        );

                    },
                    300
                );

            }

        }

    }
);