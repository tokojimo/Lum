"""Streamlit entry point for the progressive legacy-pipeline integration."""

from __future__ import annotations

from io import BytesIO

import pandas as pd
import streamlit as st
from streamlit_sortables import sort_items

from luxplate.blanks import run_blank_correction
from luxplate.crosstalk import correct_plate_crosstalk
from luxplate.export import package_figures
from luxplate.display_order import reconcile_display_order
from luxplate.figure_lifecycle import (invalidate_guided_analysis_state,
                                       validate_figure_render)
from luxplate.plotting import (build_guided_corrected_figures,
                               build_guided_crosstalk_figures, build_guided_raw_figures,
                               build_publication_figures, collect_publication_statistics,
                               directional_condition_options)
from luxplate.project import PROJECT_KEYS, export_project, import_project
from luxplate.statistics import all_pairwise_comparisons
from luxplate.varioskan import (assign_biological_pair_ids, combine_kinetic_tables,
                               inspect_workbook, parse_kinetic_workbook,
                               suggest_biological_pair_id)
from luxplate.workflow import (build_bulk_point_decisions, build_manual_decisions,
                               filter_experiment_data, run_complete_analysis)

st.set_page_config(page_title="LuxPlate Analyzer", page_icon="🧫", layout="wide")


@st.cache_data(show_spinner=False)
def cached_workbook_sheets(payload: bytes) -> tuple[str, list[str]]:
    """Inspect an uploaded workbook only once across Streamlit reruns."""
    return inspect_workbook(BytesIO(payload))


@st.cache_data(show_spinner=False)
def cached_kinetic_workbook(payload: bytes, luminescence_sheet: str) -> pd.DataFrame:
    """Parse an unchanged workbook/sheet selection only once."""
    return parse_kinetic_workbook(BytesIO(payload), luminescence_sheet)


@st.cache_data(show_spinner="Analyse en cours…")
def cached_complete_analysis(
    data: pd.DataFrame,
    decisions: pd.DataFrame,
    minimum_od: float,
    consecutive_points: int,
    growth_window_points: int,
    growth_rate_min_r_squared: float,
):
    """Reuse a complete run without caching Streamlit UI side effects.

    Progress widgets must stay outside this cached function.  Otherwise
    Streamlit records calls made through the callback and cannot replay them
    when the progress widget belongs to a later script run.
    """
    return run_complete_analysis(
        data,
        decisions,
        minimum_od=minimum_od,
        consecutive_points=consecutive_points,
        growth_window_points=growth_window_points,
        growth_rate_min_r_squared=growth_rate_min_r_squared,
    )


@st.cache_data(show_spinner="Préparation des courbes…", max_entries=12)
def cached_guided_raw_figures(data: pd.DataFrame, sample_type: str):
    """Build costly raw previews only once for each selection."""
    return build_guided_raw_figures(data, sample_type=sample_type)


@st.cache_data(show_spinner="Préparation des courbes corrigées…", max_entries=12)
def cached_guided_corrected_figures(
    data: pd.DataFrame, sample_type: str, lum_value: str = "Lum_corr"
):
    """Build blank-corrected previews only once for each selection."""
    corrected = run_blank_correction(data).corrected_data
    return build_guided_corrected_figures(
        corrected, sample_type=sample_type, lum_value=lum_value
    )


@st.cache_data(show_spinner="Préparation des courbes après cross-talk…", max_entries=12)
def cached_guided_crosstalk_figures(data: pd.DataFrame, sample_type: str):
    """Build previews of the optical correction before blank subtraction."""
    return build_guided_crosstalk_figures(data, sample_type=sample_type)


def cached_publication_figures(
    data: pd.DataFrame,
    title: str,
    families: tuple[str, ...],
    panel_by: str,
    lum_scale: str,
    metric_scale: str,
    directional_comparisons: tuple[tuple[str, str], ...],
    show_technical_replicates: bool,
    statistical_transform: str,
    alternative: str,
    width_scale: float,
    height_scale: float,
    medium_order: tuple[str, ...],
    reporter_order: tuple[str, ...],
):
    """Create fresh publication figures for the current Streamlit render.

    The expensive scientific result passed in ``data`` is already cached by
    ``cached_complete_analysis``.  Figure instances themselves must not enter
    ``st.cache_data``: Matplotlib and render/export operations mutate them and a
    later rerun could consequently receive a closed or stale canvas.
    """
    return build_publication_figures(
        data, title=title, families=families, panel_by=panel_by, lum_scale=lum_scale,
        metric_scale=metric_scale,
        directional_comparisons=directional_comparisons,
        show_technical_replicates=show_technical_replicates,
        statistical_transform=statistical_transform, alternative=alternative,
        width_scale=width_scale, height_scale=height_scale,
        medium_order=medium_order, reporter_order=reporter_order,
    )


@st.fragment
def select_display_order(media: list[str], reporters: list[str]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Edit drag-and-drop drafts and expose only explicitly validated orders."""
    specifications = (("media", "Ordre des milieux", media),
                      ("reporters", "Ordre des souches / reporters", reporters))
    drafts: dict[str, list[str]] = {}
    for name, label, present in specifications:
        validated_key = f"guided_display_order_{name}"
        draft_key = f"guided_display_order_{name}_draft"
        validated = reconcile_display_order(st.session_state.get(validated_key, []), present)
        st.session_state[validated_key] = validated
        draft_seed = reconcile_display_order(st.session_state.get(draft_key, validated), present)
        st.markdown(f"**{label}**")
        drafts[name] = sort_items(draft_seed, direction="vertical", key=draft_key)

    controls = st.columns(2)
    if controls[0].button("Valider l'ordre d'affichage", type="primary"):
        for name, _, _ in specifications:
            st.session_state[f"guided_display_order_{name}"] = drafts[name]
        st.rerun()
    if controls[1].button("Réinitialiser l'ordre"):
        for name, _, present in specifications:
            st.session_state[f"guided_display_order_{name}"] = list(present)
            st.session_state[f"guided_display_order_{name}_draft"] = list(present)
        st.rerun()
    return (tuple(st.session_state["guided_display_order_media"]),
            tuple(st.session_state["guided_display_order_reporters"]))


@st.fragment
def select_directional_comparisons(data: pd.DataFrame, *, key: str) -> tuple[tuple[str, str], ...]:
    """Build a hypothesis stack without running it until explicit validation.

    Draft edits stay inside the fragment so they do not rebuild every figure.
    Validation explicitly reruns the complete application because the return
    value is consumed below to rebuild figures with the accepted hypotheses.
    """
    conditions = directional_condition_options(data)
    if len(conditions) < 2:
        st.info("Au moins deux boîtes sont nécessaires pour définir une comparaison.")
        return ()

    stack_key = f"{key}_stack"
    validated_key = f"{key}_validated"
    valid_ids = set(conditions.values())
    if stack_key not in st.session_state:
        defaults = list(all_pairwise_comparisons(conditions.values()))
        st.session_state[stack_key] = defaults
        st.session_state[validated_key] = defaults
    stack = [tuple(pair) for pair in st.session_state.get(stack_key, [])
             if len(pair) == 2 and pair[0] in valid_ids and pair[1] in valid_ids and pair[0] != pair[1]]
    # Preserve insertion order while removing hypotheses added through more than one shortcut.
    stack = list(dict.fromkeys(stack))
    st.session_state[stack_key] = stack
    validated = [tuple(pair) for pair in st.session_state.get(validated_key, [])
                 if len(pair) == 2 and pair[0] in valid_ids and pair[1] in valid_ids
                 and pair[0] != pair[1]]
    validated = list(dict.fromkeys(validated))
    st.session_state[validated_key] = validated

    st.markdown("**Ajout manuel**")
    columns = st.columns(2)
    reference_label = columns[0].selectbox(
        "Boîte de référence (A)", list(conditions), key=f"{key}_reference",
        help="La condition dont vous testez si la moyenne est supérieure.",
    )
    comparator_labels = [label for label in conditions if label != reference_label]
    comparator_key = f"{key}_comparators"
    if comparator_key in st.session_state:
        st.session_state[comparator_key] = [
            label for label in st.session_state[comparator_key] if label in comparator_labels
        ]
    selected = columns[1].multiselect(
        "Boîte(s) à comparer (B)", comparator_labels, key=comparator_key,
        placeholder="Choisir une ou plusieurs boîtes",
        help="Un test A > B sera créé pour chaque boîte sélectionnée.",
    )
    if st.button("Ajouter à la pile", key=f"{key}_add_manual", disabled=not selected):
        additions = [(conditions[reference_label], conditions[label]) for label in selected]
        st.session_state[stack_key] = list(dict.fromkeys([*stack, *additions]))
        st.rerun(scope="fragment")

    st.markdown("**Ajouts automatiques**")
    st.caption("Générez une série d'hypothèses, puis complétez-la avec les choix manuels ci-dessus.")
    parsed = {internal: tuple(internal.split("\0", 1)) for internal in conditions.values()}
    media = list(dict.fromkeys(medium for _, medium in parsed.values()))
    controls = st.columns(2)
    if len(media) >= 2:
        medium_a = controls[0].selectbox("Milieu à gauche", media, key=f"{key}_medium_a")
        medium_b_options = [medium for medium in media if medium != medium_a]
        medium_b = controls[1].selectbox("Milieu à droite", medium_b_options, key=f"{key}_medium_b")
        same_strain = []
        strains = list(dict.fromkeys(strain for strain, _ in parsed.values()))
        for strain in strains:
            left, right = strain + "\0" + medium_a, strain + "\0" + medium_b
            if left in valid_ids and right in valid_ids:
                same_strain.append((left, right))
        if st.button(
            f"Toutes les souches : {medium_a} > {medium_b}", key=f"{key}_add_media",
            disabled=not same_strain,
        ):
            st.session_state[stack_key] = list(dict.fromkeys([*stack, *same_strain]))
            st.rerun(scope="fragment")
    else:
        st.info("Deux milieux sont nécessaires pour automatiser les comparaisons par souche.")

    p0_strains = list(dict.fromkeys(
        strain for strain, _ in parsed.values()
        if "p0" in "".join(character for character in strain.casefold() if character.isalnum())
    ))
    if p0_strains:
        control = st.selectbox("Contrôle", p0_strains, key=f"{key}_control")
        versus_control = []
        for left_id, (strain, medium) in parsed.items():
            right_id = control + "\0" + medium
            if strain != control and right_id in valid_ids:
                versus_control.append((left_id, right_id))
        if st.button(
            f"Toutes les souches > {control} dans leur milieu", key=f"{key}_add_control",
            disabled=not versus_control,
        ):
            st.session_state[stack_key] = list(dict.fromkeys([*stack, *versus_control]))
            st.rerun(scope="fragment")
    else:
        st.info("Aucune souche P0 détectée pour générer les comparaisons au contrôle.")

    stack = st.session_state[stack_key]
    labels_by_id = {internal: label for label, internal in conditions.items()}
    st.markdown(f"**Pile — {len(stack)} hypothèse{'s' if len(stack) != 1 else ''}**")
    if not stack:
        st.caption("La pile est vide : aucun test ne sera effectué.")
    else:
        for index, (left, right) in enumerate(stack):
            label_column, remove_column = st.columns([8, 1])
            label_column.write(f"{index + 1}. {labels_by_id[left]} > {labels_by_id[right]}")
            if remove_column.button("✕", key=f"{key}_remove_{index}", help="Retirer cette hypothèse"):
                st.session_state[stack_key] = stack[:index] + stack[index + 1:]
                st.rerun(scope="fragment")
        if st.button("Vider la pile", key=f"{key}_clear"):
            st.session_state[stack_key] = []
            st.rerun(scope="fragment")

    pending_changes = stack != validated
    if st.button(
        "Valider les hypothèses et lancer les tests",
        key=f"{key}_validate", type="primary", disabled=not stack,
        help="Les figures et les tests ne sont recalculés qu'après cette validation.",
    ):
        st.session_state[validated_key] = list(stack)
        # Only validation must invalidate the caller's figure gallery.  Being
        # explicit here avoids treating this as another fragment-only rerun.
        st.rerun(scope="app")
    if pending_changes:
        st.info(
            "La pile a été modifiée. Validez-la pour appliquer ces hypothèses aux tests ; "
            "les résultats affichés restent inchangés jusque-là."
        )
    elif validated:
        st.success(f"{len(validated)} hypothèse(s) validée(s) et appliquée(s).")
    return tuple(validated)


def render_guided_results(complete, base: str) -> None:
    """Render a restored or freshly completed guided analysis."""
    st.subheader("Résultats complets")
    cards = st.columns(4)
    cards[0].metric("Points exclus", len(complete.blank_correction.excluded_data))
    cards[1].metric("Points normalisés", int(complete.normalization.normalized_data["normalization_ok"].sum()))
    cards[2].metric("Séries analysées", len(complete.kinetics.series_metrics))
    cards[3].metric("Avertissements", len(complete.kinetics.warnings))
    result_tabs = st.tabs([
        "Blancs corrigés", "Normalisation", "Cinétique", "Courbes finales", "⬇ Exports",
    ])
    with result_tabs[0]:
        st.dataframe(complete.blank_correction.blank_profiles, use_container_width=True, hide_index=True)
    with result_tabs[1]:
        st.dataframe(complete.normalization.normalized_data, use_container_width=True, hide_index=True)
    with result_tabs[2]:
        st.dataframe(complete.kinetics.series_metrics, use_container_width=True, hide_index=True)
    with result_tabs[3]:
        st.caption(
            "Choisissez les figures à afficher : elles sont régénérées dès qu'une option change, "
            "sans relancer l'analyse."
        )
        guided_family_labels = {
            "Croissance corrigée": "growth",
            "Luminescence non normalisée": "corrected",
            "Double axe DO + luminescence non normalisée": "mixed",
            "Pic normalisé": "peak",
            "Pic normalisé — fold change vs P0 par milieu": "peak_fc",
            "Temps du pic normalisé": "peak_time",
            "Rapport AUC luminescence / AUC DO": "auc",
            "Rapport AUC luminescence / AUC DO — fold change vs P0 par milieu": "auc_fc",
            "Temps de doublement": "doubling",
        }
        guided_figure_labels = st.multiselect(
            "Figures finales", list(guided_family_labels),
            default=[
                "Croissance corrigée",
                "Luminescence non normalisée",
                "Double axe DO + luminescence non normalisée",
                "Rapport AUC luminescence / AUC DO",
            ], key="guided_figure_families",
        )
        st.markdown("#### Affichage")
        guided_options = st.columns(4)
        guided_panel_label = guided_options[0].selectbox(
            "Organisation des panneaux", ["Panneaux par milieu", "Panneaux par souche"],
            key="guided_figure_panels",
        )
        guided_lum_label = guided_options[1].selectbox(
            "Échelle Y des courbes", ["Linéaire", "Logarithmique (base 10)"],
            index=0, key="guided_figure_lum_scale",
            help=("Ce réglage modifie uniquement l’affichage de la figure. "
                  "Il ne transforme pas les valeurs utilisées pour les tests statistiques."),
        )
        guided_metric_label = guided_options[2].selectbox(
            "Échelle Y des boîtes à moustaches", ["Linéaire", "Logarithmique (base 10)"],
            index=1, key="guided_figure_metric_scale",
            help=("Ce réglage graphique est indépendant de la transformation statistique."),
        )
        guided_show_technical = guided_options[3].checkbox(
            "Afficher les réplicats techniques",
            value=True,
            key="guided_show_technical_replicates",
            help=("Ajoute un petit point par puits technique dans les figures de synthèse, "
                  "quel que soit le nombre de réplicats."),
        )
        st.markdown("##### Ordre d'affichage")
        available_conditions = directional_condition_options(
            complete.normalization.normalized_data
        ).values()
        available_pairs = [condition.split("\0", 1) for condition in available_conditions]
        natural_reporters = list(dict.fromkeys(pair[0] for pair in available_pairs))
        natural_media = list(dict.fromkeys(pair[1] for pair in available_pairs))
        guided_medium_order, guided_reporter_order = select_display_order(
            natural_media, natural_reporters
        )
        st.markdown("#### Dimensions des figures")
        st.caption(
            "Ajustez séparément la longueur des axes X et Y. Ces dimensions sont aussi "
            "appliquées aux fichiers téléchargés."
        )
        dimension_options = st.columns(2)
        guided_width_percent = dimension_options[0].slider(
            "Largeur (axe X)", min_value=50, max_value=200, value=100, step=10,
            format="%d %%", key="guided_figure_width_percent",
        )
        guided_height_percent = dimension_options[1].slider(
            "Hauteur (axe Y)", min_value=50, max_value=200, value=100, step=10,
            format="%d %%", key="guided_figure_height_percent",
            help="Augmentez cette valeur pour obtenir une figure plus longue verticalement.",
        )
        st.markdown("#### Statistiques")
        statistics_options = st.columns(3)
        guided_test_label = statistics_options[0].selectbox(
            "Test", ["t-test apparié bilatéral", "t-test apparié unilatéral (A > B)"],
            key="guided_statistical_test",
        )
        guided_transform_label = statistics_options[1].selectbox(
            "Transformation utilisée pour le test", ["log10", "Valeurs brutes"],
            key="guided_statistical_transform",
            help=("Cette transformation est appliquée uniquement aux valeurs utilisées pour le test "
                  "statistique. Elle est indépendante de l’échelle utilisée pour afficher la figure."),
        )
        statistics_options[2].selectbox(
            "Correction des comparaisons multiples", ["Holm"], disabled=True,
            key="guided_multiple_testing",
        )
        guided_transform = "log10" if guided_transform_label == "log10" else "none"
        guided_alternative = ("two-sided" if "bilatéral" in guided_test_label else "greater")
        with st.expander("Comparaisons statistiques", expanded=False):
            st.caption(
                "Sélectionnez uniquement les hypothèses définies avant de consulter les résultats. "
                + ("Chaque choix compare les deux conditions dans les deux sens."
                   if guided_alternative == "two-sided" else
                   "Chaque choix teste la condition à gauche comme supérieure à celle de droite.")
            )
            guided_selected_comparisons = select_directional_comparisons(
                complete.normalization.normalized_data,
                key="guided_directional_comparisons",
            )
        guided_figures = []
        if not guided_figure_labels:
            st.info("Sélectionnez au moins une figure finale.")
        else:
            guided_figures = cached_publication_figures(
                complete.normalization.normalized_data,
                title="Analyse complète",
                families=tuple(guided_family_labels[label] for label in guided_figure_labels),
                panel_by="Groupe" if guided_panel_label.endswith("milieu") else "souche",
                lum_scale="log" if guided_lum_label == "Logarithmique (base 10)" else "linear",
                metric_scale="log" if guided_metric_label == "Logarithmique (base 10)" else "linear",
                directional_comparisons=guided_selected_comparisons,
                show_technical_replicates=guided_show_technical,
                statistical_transform=guided_transform, alternative=guided_alternative,
                width_scale=guided_width_percent / 100,
                height_scale=guided_height_percent / 100,
                medium_order=guided_medium_order,
                reporter_order=guided_reporter_order,
            )
            assert guided_figures, "La construction n'a produit aucune figure."
            statistical_results = collect_publication_statistics(guided_figures)
            def display_condition(item):
                parts = item if isinstance(item, tuple) else str(item).split("\0", 1)
                return " · ".join(map(str, parts))
            for guided_name, guided_figure in guided_figures:
                st.subheader(guided_name.replace("_", " ").title())
                render_diagnostic = validate_figure_render(guided_figure)
                # Kept out of the normal interface, but available in server
                # logs when diagnosing a deployment-specific renderer issue.
                print(f"LuxPlate figure {guided_name}: {render_diagnostic}")
                # Be explicit: Streamlit must not clear this mutable Figure;
                # it is still needed below for statistics and export.
                st.pyplot(guided_figure, use_container_width=True, clear_figure=False)
                statistics = getattr(guided_figure, "_luxplate_statistics", pd.DataFrame())
                if not statistics.empty:
                    visible_statistics = statistics.copy()
                    for column in ("condition_1", "condition_2"):
                        visible_statistics[column] = visible_statistics[column].map(display_condition)
                    st.dataframe(
                        visible_statistics[["condition_1", "condition_2", "test", "alternative",
                                    "statistical_transform", "n_pairs", "p_raw", "p_adjusted",
                                    "significance", "calculation_status",
                                    "non_calculable_reason"]].rename(columns={
                            "condition_1": "Condition A", "condition_2": "Condition B",
                            "test": "Test", "alternative": "Alternative",
                            "statistical_transform": "Transformation",
                            "n_pairs": "Paires biologiques", "p_raw": "p brute",
                            "p_adjusted": "p ajustée Holm",
                            "significance": "Résultat",
                            "calculation_status": "Statut",
                            "non_calculable_reason": "Raison si non calculable",
                        }),
                        use_container_width=True, hide_index=True,
                    )
                diagnostics = getattr(guided_figure, "_luxplate_statistical_diagnostics", [])
                if diagnostics:
                    with st.expander("Diagnostic statistique", expanded=False):
                        for diagnostic in diagnostics:
                            left = diagnostic.get("condition_a") or diagnostic["requested_left"]
                            right = diagnostic.get("condition_b") or diagnostic["requested_right"]
                            st.markdown(f"**Condition A :** {display_condition(left)}")
                            st.markdown(f"**Condition B :** {display_condition(right)}")
                            st.markdown(
                                "**Identité biologique utilisée :** "
                                + " + ".join(diagnostic["identity_columns"])
                            )
                            st.dataframe(diagnostic["pair_table"], use_container_width=True, hide_index=True)
                            used = (diagnostic["positive_pairs"] if guided_transform == "log10"
                                    else diagnostic["finite_pairs"])
                            st.write(f"Paires complètes : {diagnostic['complete_pairs']}")
                            st.write(f"Paires utilisées : {used}")
                            st.write(
                                "Paires éliminées (valeur non finie) : ",
                                diagnostic["complete_pairs"] - diagnostic["finite_pairs"],
                            )
                            if guided_transform == "log10":
                                st.write(
                                    "Paires éliminées (valeur ≤ 0) : ",
                                    diagnostic["finite_pairs"] - diagnostic["positive_pairs"],
                                )
                            if diagnostic["only_a"]:
                                st.write("Expériences présentes uniquement dans A :", diagnostic["only_a"])
                            if diagnostic["only_b"]:
                                st.write("Expériences présentes uniquement dans B :", diagnostic["only_b"])
                            st.divider()
            if not statistical_results.empty:
                st.download_button(
                    "Exporter les statistiques (.csv)",
                    statistical_results.to_csv(index=False).encode("utf-8-sig"),
                    f"{base}_statistiques.csv", "text/csv",
                    key="guided_statistics_export",
                )
            st.caption(
                "Les temps séparés de moins d'une minute sont alignés avant le tracé. Les puits "
                "techniques sont moyennés dans chaque réplicat biologique, puis les courbes montrent "
                "la moyenne biologique ± SD. Normalized "
                "luminescence is blank-corrected RLU divided by blank-corrected OD600. "
                "Summary boxplots show independent biological experiments and their exact mean; "
                "Quand leur affichage est activé, les petits points sont les réplicats techniques "
                "et les grands points les moyennes biologiques. "
                f"Les seules statistiques affichées sont les hypothèses sélectionnées : {guided_test_label} "
                f"— {guided_transform_label} des moyennes biologiques. Chaque comparaison "
                "planifiée 2 à 2 utilise sa p-value brute pour les étoiles ; la correction de "
                "Holm reste disponible dans le tableau statistique exporté."
            )
    with result_tabs[4]:
        st.subheader("Exporter les résultats complets")
        st.caption(
            "Téléchargez directement les tableaux et les figures produits par "
            "l'analyse guidée."
        )
        exports = st.columns(3)
        exports[0].download_button("Données finales (.csv)", complete.normalization.normalized_data.to_csv(
            index=False).encode("utf-8-sig"), f"{base}_donnees_finales.csv", "text/csv",
            type="primary")
        exports[1].download_button("Métriques cinétiques (.csv)", complete.kinetics.series_metrics.to_csv(
            index=False).encode("utf-8-sig"), f"{base}_metriques_cinetiques.csv", "text/csv")
        exports[2].download_button("Décisions d'exclusion (.csv)", st.session_state["guided_decisions"].to_csv(
            index=False).encode("utf-8-sig"), f"{base}_decisions_exclusion.csv", "text/csv")
        st.divider()
        st.subheader("Figures finales")
        if not guided_figures:
            st.info("Sélectionnez au moins une figure dans l'onglet « Courbes finales ».")
        else:
            guided_export_dpi = st.select_slider(
                "Qualité des fichiers PNG et TIF (dpi)",
                options=[150, 300, 600], value=300, key="guided_export_dpi",
                help="Le SVG reste vectoriel, quelle que soit cette valeur.",
            )
            with st.spinner("Préparation des figures à télécharger…"):
                guided_rendered, guided_archive = package_figures(
                    guided_figures, dpi=guided_export_dpi,
                )
            st.download_button(
                "Toutes les figures (ZIP)", guided_archive,
                file_name=f"{base}_figures.zip", mime="application/zip", type="primary",
                key="guided_figures_zip",
            )
            for item in guided_rendered:
                st.markdown(f"**{item.name.replace('_', ' ').title()}**")
                png_export, tif_export, svg_export = st.columns(3)
                png_export.download_button(
                    f"PNG · {guided_export_dpi} dpi", item.png, f"{item.name}.png",
                    "image/png", key=f"guided_png_{item.name}",
                )
                tif_export.download_button(
                    f"TIF · {guided_export_dpi} dpi", item.tiff, f"{item.name}.tif",
                    "image/tiff", key=f"guided_tif_{item.name}",
                )
                svg_export.download_button(
                    "SVG · vectoriel", item.svg, f"{item.name}.svg",
                    "image/svg+xml", key=f"guided_svg_{item.name}",
                )

st.title("LuxPlate Analyzer")
st.caption("Du classeur Varioskan brut aux données contrôlées, analysées et visualisées.")

with st.sidebar:
    st.header("💾 Projet")
    st.caption("Sauvegardez l'analyse pour la reprendre après avoir fermé votre session.")
    project_upload = st.file_uploader(
        "Importer un projet LuxPlate (.luxplate)", type=["luxplate"], key="project_upload",
    )
    if project_upload is not None:
        payload = project_upload.getvalue()
        project_identity = (project_upload.name, len(payload), hash(payload))
        if st.session_state.get("loaded_project_identity") != project_identity:
            try:
                restored_state = import_project(payload)
            except ValueError as error:
                st.error(f"Import du projet impossible : {error}")
            else:
                # Importing replaces the current analysis instead of merging
                # it with values left by a previously opened project.
                for key in PROJECT_KEYS:
                    st.session_state.pop(key, None)
                for key, value in restored_state.items():
                    st.session_state[key] = value
                st.session_state["loaded_project_identity"] = project_identity
                st.success("Projet restauré. Vous pouvez reprendre l'analyse dans les onglets.")
                st.rerun()
    if "long_data" in st.session_state or "guided_complete_result" in st.session_state:
        project_name = st.session_state.get("source_name", "analyse")
        st.download_button(
            "Exporter le projet", export_project(dict(st.session_state)),
            file_name=f"{project_name}.luxplate", mime="application/zip", type="primary",
            help="Inclut les données importées, décisions QC et résultats de calcul.",
        )
    else:
        st.info("Importez des données pour activer l'export du projet.")

guided_tab = st.container()

with guided_tab:
    st.header("Analyse guidée d'un classeur réel")
    st.write(
        "Déposez un ou plusieurs fichiers, vérifiez les souches détectées, retirez si nécessaire "
        "des points ou des courbes, puis lancez tout le calcul en une fois."
    )
    guided_uploads = st.file_uploader(
        "Déposer un ou plusieurs classeurs Varioskan (.xlsx ou .xlsm)", type=["xlsx", "xlsm"],
        accept_multiple_files=True, key="guided_upload"
    )
    if not guided_uploads:
        if "guided_complete_result" in st.session_state:
            st.success("Projet chargé : l'analyse est restaurée au niveau où elle a été sauvegardée.")
            render_guided_results(
                st.session_state["guided_complete_result"],
                st.session_state.get("source_name", "analyse"),
            )
        elif "long_data" in st.session_state:
            st.success("Projet chargé : les données préparées ont été restaurées.")
            st.info("Ajoutez les classeurs d'origine pour relancer ou modifier l'analyse.")
        else:
            st.info("Les fichiers restent traités localement par l'application.")
    else:
        try:
            guided_inputs = []
            guided_identity = []
            guided_import_progress = st.progress(0, text="Lecture des classeurs…")
            for file_index, upload in enumerate(guided_uploads):
                payload = upload.getvalue()
                _, lum_sheets = cached_workbook_sheets(payload)
                lum = st.selectbox(f"Luminescence — {upload.name}", lum_sheets,
                                   key=f"guided_lum_{file_index}_{upload.name}")
                guided_inputs.append((upload.name, cached_kinetic_workbook(payload, lum)))
                guided_identity.append((upload.name, hash(payload), lum))
                guided_import_progress.progress(
                    int((file_index + 1) / len(guided_uploads) * 90),
                    text=f"Classeur {file_index + 1}/{len(guided_uploads)} lu",
                )
            guided_data = combine_kinetic_tables(guided_inputs)
            guided_import_progress.progress(100, text="Chargement terminé")
            guided_import_progress.empty()
        except Exception as error:
            st.error(f"Import impossible : {error}")
        else:
            selection_identity = tuple(guided_identity)
            if (st.session_state.get("guided_selection_identity") != selection_identity
                    or "guided_pair_mapping" not in st.session_state):
                # A newly uploaded workbook must not silently reuse draft or
                # validated choices belonging to the preceding dataset.
                for key in ("guided_media", "guided_strains", "guided_crosstalk"):
                    st.session_state.pop(key, None)
                st.session_state.pop("guided_validated_selection", None)
                st.session_state.pop("guided_validated_pair_mapping", None)
                st.session_state["guided_pair_mapping"] = pd.DataFrame({
                    "experience": list(dict.fromkeys(guided_data["experience"])),
                    "biological_pair_id": [
                        suggest_biological_pair_id(item)
                        for item in dict.fromkeys(guided_data["experience"])
                    ],
                })
                st.session_state["guided_selection_identity"] = selection_identity

            st.subheader("Correspondance des réplicats biologiques")
            st.caption(
                "Lum propose une correspondance à partir des marqueurs rep1/rep2/rep3. "
                "Vérifiez ou modifiez chaque valeur, puis validez-la explicitement. "
                "L'ordre et le nombre des fichiers ne créent jamais de paires automatiquement."
            )
            pair_mapping = st.data_editor(
                st.session_state["guided_pair_mapping"],
                disabled=["experience"], hide_index=True, use_container_width=True,
                column_config={
                    "experience": "Fichier / expérience",
                    "biological_pair_id": "Réplicat biologique",
                }, key="guided_pair_mapping_editor",
            )
            if st.button("Valider les correspondances biologiques", type="primary"):
                try:
                    assign_biological_pair_ids(guided_data, pair_mapping)
                except ValueError as error:
                    st.error(str(error))
                else:
                    st.session_state["guided_pair_mapping"] = pair_mapping.copy()
                    st.session_state["guided_validated_pair_mapping"] = {
                        "identity": selection_identity,
                        "mapping": pair_mapping.copy(),
                    }

            validated_pairs = st.session_state.get("guided_validated_pair_mapping")
            if not validated_pairs or validated_pairs.get("identity") != selection_identity:
                st.info("Vérifiez puis validez les correspondances biologiques pour continuer.")
                st.stop()
            try:
                guided_data = assign_biological_pair_ids(
                    guided_data, validated_pairs["mapping"]
                )
            except ValueError as error:
                st.error(str(error))
                st.stop()

            strain_options = sorted(
                guided_data.loc[guided_data["type"].eq("souche"), "souche"].unique()
            )
            medium_options = sorted(
                guided_data.loc[guided_data["type"].eq("souche"), "Groupe"].unique()
            )
            st.subheader("Sélection de l'analyse")
            st.caption(
                "Préparez votre sélection librement : aucun calcul ni graphique ne sera "
                "relancé avant la validation."
            )
            with st.form("guided_selection_form"):
                guided_crosstalk_draft = st.checkbox(
                    "Corriger le cross-talk de luminescence — Mauri Dbest", value=True,
                    key="guided_crosstalk",
                    help=("Déconvolution matricielle 96×96 utilisant le kernel Dbest calibré "
                          "sur une plaque mono-source E06."),
                )
                selection_columns = st.columns(2)
                guided_media_draft = selection_columns[0].multiselect(
                    "Milieux à analyser", medium_options, default=medium_options,
                    key="guided_media",
                )
                guided_strains_draft = selection_columns[1].multiselect(
                    "Souches à analyser", strain_options, default=strain_options,
                    key="guided_strains",
                )
                selection_submitted = st.form_submit_button(
                    "Valider la sélection", type="primary",
                    disabled=not medium_options or not strain_options,
                    help="L'analyse des milieux et souches choisis commence uniquement ici.",
                )

            if selection_submitted:
                st.session_state["guided_validated_selection"] = {
                    "identity": selection_identity,
                    "media": tuple(guided_media_draft),
                    "strains": tuple(guided_strains_draft),
                    "crosstalk": guided_crosstalk_draft,
                }

            validated_selection = st.session_state.get("guided_validated_selection")
            if not validated_selection or validated_selection.get("identity") != selection_identity:
                st.info(
                    "Choisissez les milieux et les souches, puis cliquez sur « Valider la "
                    "sélection » pour démarrer la préparation de l'analyse."
                )
                st.stop()

            guided_media = list(validated_selection["media"])
            guided_strains = list(validated_selection["strains"])
            guided_crosstalk = bool(validated_selection["crosstalk"])
            if guided_crosstalk:
                try:
                    guided_data = correct_plate_crosstalk(guided_data)
                except ValueError as error:
                    st.error(f"Correction du cross-talk impossible : {error}")
                    st.stop()
                st.caption("Cross-talk corrigé avec MAURI_E06_BEST.")
            st.success(
                f"Sélection validée : {len(guided_media)} milieu(x) et "
                f"{len(guided_strains)} souche(s)."
            )
            try:
                guided_selected = filter_experiment_data(guided_data, guided_strains, guided_media)
            except ValueError as error:
                st.warning(str(error))
            else:
                guided_signature = (
                    tuple(guided_identity), tuple(guided_media), tuple(guided_strains), guided_crosstalk,
                    tuple(validated_pairs["mapping"].itertuples(index=False, name=None)),
                )
                if st.session_state.get("guided_signature") != guided_signature:
                    invalidate_guided_analysis_state(st.session_state)
                    # A new analysis always starts with the inexpensive blank
                    # preview.  The other previews are deliberately lazy and
                    # must be requested again for the new data selection.
                    st.session_state["guided_signature"] = guided_signature
                blanks = guided_selected.loc[guided_selected["type"].eq("blanc")]
                if blanks.empty:
                    st.error("Aucun blanc n'a été détecté pour les milieux sélectionnés.")
                else:
                    st.success(
                        f"{len(guided_strains)} souche(s), {len(guided_uploads)} fichier(s), "
                        f"{guided_selected['sample_header'].nunique()} courbe(s) et "
                        f"{blanks['sample_header'].nunique()} blanc(s) détectés."
                    )
                tab_names = ["Courbes des blancs", "Courbes des échantillons"]
                if guided_crosstalk:
                    tab_names.append("Après correction du cross-talk")
                tab_names.extend([
                    "Blancs déconvolués", "Résidu des blancs", "Luminescence corrigée"
                ])
                selected_figure = st.radio(
                    "Figure à afficher",
                    tab_names,
                    horizontal=True,
                    key="guided_figure_view",
                    help=("Seules les courbes des blancs sont préparées au lancement. "
                          "Les autres figures sont générées à la demande puis conservées "
                          "en cache pendant la session."),
                )
                if selected_figure == "Courbes des blancs":
                    st.caption(
                        "Une ligne par expérience indépendante (réplicat biologique) ; les courbes d'une "
                        "même ligne sont les réplicats techniques. Les échelles sont identiques entre les lignes."
                    )
                    for _, figure in cached_guided_raw_figures(guided_selected, sample_type="blanc"):
                        st.pyplot(figure, use_container_width=True)
                elif selected_figure == "Courbes des échantillons":
                    st.caption(
                        "Une figure par souche et milieu, avec une ligne par fichier Excel (réplicat biologique) "
                        "et toutes ses courbes techniques. Les maxima OD600 et RLU sont communs aux lignes."
                    )
                    for _, figure in cached_guided_raw_figures(guided_selected, sample_type="souche"):
                        st.pyplot(figure, use_container_width=True)
                elif selected_figure == "Après correction du cross-talk":
                    st.caption(
                        "Luminescence après correction du cross-talk et avant soustraction "
                        "des blancs. Cette vue permet d'évaluer directement le modèle optique."
                    )
                    for _, figure in cached_guided_crosstalk_figures(
                        guided_selected, sample_type="blanc"
                    ):
                        st.pyplot(figure, use_container_width=True)
                elif selected_figure == "Blancs déconvolués":
                    st.caption(
                        "Niveau absolu des blancs après correction optique. Il n'est pas recentré : "
                        "ces courbes permettent d'évaluer directement la déconvolution du cross-talk."
                    )
                    for _, figure in cached_guided_corrected_figures(
                        guided_selected, sample_type="blanc"
                    ):
                        st.pyplot(figure, use_container_width=True)
                elif selected_figure == "Résidu des blancs":
                    st.caption(
                        "Contrôle qualité uniquement : Lum_blank_residual soustrait, à chaque temps, "
                        "la moyenne des blancs du même milieu. Sa moyenne vaut zéro par construction."
                    )
                    for _, figure in cached_guided_corrected_figures(
                        guided_selected, sample_type="blanc", lum_value="Lum_blank_residual"
                    ):
                        st.pyplot(figure, use_container_width=True)
                elif selected_figure == "Luminescence corrigée":
                    st.caption(
                        "Courbes des échantillons après soustraction, à chaque temps, de la moyenne "
                        "des blancs du même milieu. La densité optique corrigée est affichée à gauche "
                        "et la luminescence corrigée à droite."
                    )
                    for _, figure in cached_guided_corrected_figures(
                        guided_selected, sample_type="souche"
                    ):
                        st.pyplot(figure, use_container_width=True)

                st.subheader("Points et courbes à supprimer")
                st.caption(
                    "Cochez les lignes à supprimer. Une suppression de point retire ensemble la DO et la "
                    "luminescence mesurées à ce temps ; le fichier source n'est jamais modifié."
                )
                editable_points = guided_selected.reset_index(names="source_index").copy()
                editable_points.insert(0, "Supprimer", False)
                edited_points = st.data_editor(
                    editable_points[["Supprimer", "source_index", "type", "souche", "Groupe", "sample_header",
                                     "puits", "temps_h", "DO_brute", "Lum_brute"]],
                    disabled=["source_index", "type", "souche", "Groupe", "sample_header", "puits",
                              "temps_h", "DO_brute", "Lum_brute"],
                    hide_index=True, use_container_width=True, key="guided_point_editor",
                )
                curve_headers = guided_selected["sample_header"].drop_duplicates().tolist()
                removed_series = st.multiselect("Courbes entières à supprimer", curve_headers)

                with st.expander("Exclusion simple par expérience et par temps", expanded=True):
                    st.caption(
                        "Alternative au grand tableau : choisissez une expérience, un ou plusieurs temps, "
                        "puis excluez ces points de toutes les courbes ou seulement des blancs."
                    )
                    experience_column = "experience" if "experience" in guided_selected else None
                    if experience_column is None:
                        st.info("Ce mode devient disponible lorsque plusieurs fichiers sont importés.")
                        simple_enabled = False
                        simple_experience = ""
                        simple_times = []
                        simple_type = "all"
                    else:
                        simple_enabled = st.checkbox("Activer cette exclusion groupée")
                        simple_experience = st.selectbox(
                            "Expérience", guided_selected[experience_column].drop_duplicates().tolist()
                        )
                        simple_subset = guided_selected.loc[
                            guided_selected[experience_column].astype(str).eq(str(simple_experience))
                        ]
                        time_options = sorted(simple_subset["temps_h"].dropna().astype(float).unique())
                        simple_times = st.multiselect(
                            "Temps à exclure (h)", time_options,
                            default=time_options[:1],
                            format_func=lambda value: f"{value:g} h",
                        )
                        simple_type_label = st.radio(
                            "Courbes concernées", ["Toutes", "Blancs seulement", "Échantillons seulement"],
                            horizontal=True,
                        )
                        simple_type = {"Toutes": "all", "Blancs seulement": "blanc",
                                       "Échantillons seulement": "souche"}[simple_type_label]

                with st.expander("Paramètres de calcul", expanded=False):
                    p1, p2, p3, p4 = st.columns(4)
                    guided_min_od = p1.number_input(
                        "DO minimale", min_value=0.0, value=0.05, step=0.01, key="guided_min_od",
                    )
                    guided_consecutive = p2.number_input(
                        "Points consécutifs", min_value=1, value=3, step=1, key="guided_consecutive",
                    )
                    guided_window = p3.number_input(
                        "Points par fenêtre", min_value=2, value=3, step=1, key="guided_window",
                    )
                    guided_r2 = p4.number_input(
                        "R² minimal", min_value=0.0, max_value=1.0, value=0.0, step=0.05,
                        key="guided_r2",
                    )

                if st.button("Lancer toute l'analyse", type="primary", disabled=blanks.empty):
                    removed_points = edited_points.loc[edited_points["Supprimer"], "source_index"].astype(int).tolist()
                    manual_decisions = build_manual_decisions(guided_selected, removed_points, removed_series)
                    if simple_enabled:
                        bulk_decisions = build_bulk_point_decisions(
                            guided_selected, experience=simple_experience,
                            times=simple_times, sample_type=simple_type,
                        )
                        manual_decisions = pd.concat(
                            [manual_decisions, bulk_decisions], ignore_index=True
                        ).drop_duplicates("decision_id")
                    try:
                        analysis_progress = st.progress(0, text="Démarrage de l'analyse…")
                        complete = cached_complete_analysis(
                            guided_selected, manual_decisions, float(guided_min_od),
                            int(guided_consecutive), int(guided_window), float(guided_r2),
                        )
                        # Keep all progress-widget calls outside the cached function:
                        # cached Streamlit effects cannot safely reference a layout
                        # block created during another script run.
                        analysis_progress.progress(100, text="Analyse terminée")
                    except ValueError as error:
                        st.error(f"Analyse impossible : {error}")
                    else:
                        analysis_progress.empty()
                        st.session_state["guided_complete_result"] = complete
                        st.session_state["guided_decisions"] = manual_decisions
                        st.session_state["long_data"] = guided_selected.copy(deep=True)
                        st.session_state["source_name"] = (
                            guided_uploads[0].name.rsplit(".", 1)[0]
                            if len(guided_uploads) == 1 else "analyse_multi_fichiers"
                        )

                if "guided_complete_result" in st.session_state:
                    render_guided_results(
                        st.session_state["guided_complete_result"],
                        st.session_state.get("source_name", "analyse"),
                    )
