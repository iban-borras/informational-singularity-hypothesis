#!/usr/bin/env python3
"""
🚀 Sistema d'Agents per la Hipòtesi de Singularitat Informacional (HSI)

Aquest és el punt d'entrada principal per executar el sistema complet d'agents
que treballen sobre les seqüències Φ per detectar patrons Pₖ i inferir regles ωₖ.

Flux del sistema:
1. Generador Φ (Nivell 0): Genera seqüències binàries primitives
2. Detector de Patrons: Identifica patrons recurrents Pₖ(Φ)
3. Inferidor de Regles: Dedueix regles emergents ωₖ
4. Validador: Verifica la consistència i robustesa de les regles

Autor: Iban Borràs amb col·laboració d'Augment Agent
Data: Gener 2025
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any, List

# Imports dels mòduls del projecte
from level0 import simulate_phi
from agents import PatternDetector, RuleInferer, Validator
from metrics import create_pattern_report, create_rule_report
from utils import create_summary_dashboard

# Configuració per defecte
DEFAULT_CONFIG = {
    "level0_generation": {
        "max_iterations": 16,
        "collapse_rule": "AND",
        "collapse_depth": 1,
        "save_snapshots": True,
        "use_compression": True,
        "memory_threshold": 10**8
    },
    "pattern_detection": {
        "min_pattern_length": 3,
        "max_pattern_length": 20,
        "min_occurrences": 2,
        "similarity_threshold": 0.8
    },
    "rule_inference": {
        "context_window": 5,
        "min_rule_confidence": 0.7,
        "max_rule_complexity": 10
    },
    "validation": {
        "validation_split": 0.3,
        "min_validation_score": 0.6,
        "stability_threshold": 0.1,
        "cross_validation_folds": 5
    },
    "output": {
        "save_results": True,
        "create_visualizations": True,
        "verbose": True
    }
}


def load_config(config_path: str = "config.json") -> Dict[str, Any]:
    """Carrega la configuració des d'un fitxer JSON, creant-lo si no existeix."""
    config_file = Path(config_path)

    if config_file.exists():
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            print(f"📋 Configuració carregada des de {config_path}")
            return config
        except (json.JSONDecodeError, Exception) as e:
            print(f"⚠️  Error llegint {config_path}: {e}")
            print("📋 Usant configuració per defecte")
    else:
        # Crear config.json amb valors per defecte
        print(f"📋 Creant {config_path} amb configuració per defecte...")
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(DEFAULT_CONFIG, f, indent=2, ensure_ascii=False)
        print(f"✅ Fitxer {config_path} creat. Pots editar-lo per personalitzar l'experiment.")

    return DEFAULT_CONFIG.copy()


def setup_output_directories(base_dir: str = "results") -> Dict[str, Path]:
    """Crea els directoris de sortida necessaris."""
    base_path = Path(base_dir)
    
    directories = {
        "base": base_path,
        "phi": base_path / "phi_sequences",
        "patterns": base_path / "patterns",
        "rules": base_path / "rules",
        "validation": base_path / "validation",
        "visualizations": base_path / "visualizations",
        "reports": base_path / "reports"
    }
    
    for dir_path in directories.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Directoris de sortida creats a {base_path}")
    return directories


def run_phi_generation(config: Dict[str, Any], output_dirs: Dict[str, Path]) -> tuple:
    """Executa la generació de seqüències Φ."""
    print("\n🔧 FASE 1: Generació de Seqüències Φ (Nivell 0)")
    print("=" * 60)

    phi_config = config["level0_generation"]
    
    # Generar seqüència principal
    phi_sequence, snapshots, metadata = simulate_phi(
        max_iterations=phi_config["max_iterations"],
        collapse_rule=phi_config["collapse_rule"],
        collapse_depth=phi_config["collapse_depth"],
        save_snapshots=phi_config["save_snapshots"],
        output_dir=str(output_dirs["phi"]),
        use_compression=phi_config.get("use_compression", True),
        memory_threshold=phi_config.get("memory_threshold", 10**8)
    )
    
    print(f"✅ Seqüència Φ generada: {len(phi_sequence):,} bits")
    print(f"   Iteracions completades: {metadata['iterations_completed']}")

    # Mostrar informació de compressió si s'ha usat
    if metadata.get('compression_activated'):
        comp_summary = metadata.get('compression_summary', {})
        print(f"   🗜️  Compressió activada: {comp_summary.get('compressed_snapshots', 0)} snapshots")
        print(f"   💾 Mida total emmagatzemada: {comp_summary.get('total_size_mb', 0):.2f} MB")

    if len(phi_sequence) > 100:
        from level0.generator import _estimate_fractal_dimension
        fractal_dim = _estimate_fractal_dimension(phi_sequence)
        print(f"   Dimensió fractal estimada: {fractal_dim:.4f}")
        print(f"   Diferència amb φ (1.618): {abs(fractal_dim - 1.618):.4f}")
    
    # Generar seqüències addicionals per validació
    additional_sequences = []
    num_additional = 3
    
    print(f"\n🔄 Generant {num_additional} seqüències addicionals per validació...")
    
    for i in range(num_additional):
        # Variar lleugerament els paràmetres per diversitat
        varied_iterations = phi_config["max_iterations"] + (i - 1) * 2
        varied_iterations = max(8, varied_iterations)
        
        add_phi, _, add_metadata = simulate_phi(
            max_iterations=varied_iterations,
            collapse_rule=phi_config["collapse_rule"],
            collapse_depth=phi_config["collapse_depth"],
            save_snapshots=False
        )
        
        additional_sequences.append(add_phi)
        print(f"   Seqüència {i+1}: {len(add_phi)} bits")
    
    return phi_sequence, snapshots, metadata, additional_sequences


def run_pattern_detection(phi_sequence: str, config: Dict[str, Any], 
                         output_dirs: Dict[str, Path]) -> List[Dict[str, Any]]:
    """Executa la detecció de patrons."""
    print("\n🔍 FASE 2: Detecció de Patrons Pₖ")
    print("=" * 60)
    
    pattern_config = config["pattern_detection"]
    
    # Crear i configurar detector
    detector = PatternDetector(
        min_pattern_length=pattern_config["min_pattern_length"],
        max_pattern_length=pattern_config["max_pattern_length"],
        min_occurrences=pattern_config["min_occurrences"],
        similarity_threshold=pattern_config["similarity_threshold"]
    )
    
    # Detectar patrons
    patterns = detector.detect_patterns(phi_sequence)
    
    print(f"✅ Patrons detectats: {len(patterns)}")
    
    # Mostrar resum
    summary = detector.get_pattern_summary()
    if summary["total_patterns"] > 0:
        print(f"   Longitud mitjana: {summary['average_length']:.1f}")
        print(f"   Recurrència mitjana: {summary['average_recurrence']:.1f}")
        print(f"   Mètodes utilitzats: {', '.join(summary['methods_used'])}")
        
        # Mostrar top patrons
        print("\n📊 Top 3 patrons per recurrència:")
        for i, pattern in enumerate(summary["top_patterns"][:3]):
            print(f"   {i+1}. '{pattern['pattern_data']}' - {pattern['recurrence']} ocurrències")
    
    # Guardar resultats
    if config["output"]["save_results"]:
        detector.save_patterns(output_dirs["patterns"] / "detected_patterns.json")
        
        # Crear informes detallats
        print("\n📋 Generant informes de patrons...")
        pattern_reports = []
        
        for pattern in patterns[:10]:  # Limitar a 10 per rendiment
            report = create_pattern_report(pattern, len(phi_sequence))
            pattern_reports.append(report)
        
        with open(output_dirs["reports"] / "pattern_reports.json", 'w') as f:
            json.dump(pattern_reports, f, indent=2)
        
        print(f"   Informes guardats per {len(pattern_reports)} patrons")
    
    return patterns


def run_rule_inference(patterns: List[Dict[str, Any]], phi_sequence: str,
                      config: Dict[str, Any], output_dirs: Dict[str, Path]) -> List[Dict[str, Any]]:
    """Executa la inferència de regles."""
    print("\n🧠 FASE 3: Inferència de Regles ωₖ")
    print("=" * 60)
    
    rule_config = config["rule_inference"]
    
    # Crear i configurar inferidor
    inferer = RuleInferer(
        context_window=rule_config["context_window"],
        min_rule_confidence=rule_config["min_rule_confidence"],
        max_rule_complexity=rule_config["max_rule_complexity"]
    )
    
    # Inferir regles
    rules = inferer.infer_rules(patterns, phi_sequence)
    
    print(f"✅ Regles inferides: {len(rules)}")
    
    # Mostrar resum
    summary = inferer.get_rules_summary()
    if summary["total_rules"] > 0:
        print(f"   Confiança mitjana: {summary['average_confidence']:.3f}")
        print(f"   Precisió mitjana: {summary['average_precision']:.3f}")
        print(f"   Complexitat mitjana: {summary['average_complexity']:.3f}")
        
        # Mostrar distribució per tipus
        print(f"   Tipus de regles: {summary['rule_types']}")
        
        # Mostrar top regles
        print("\n📊 Top 3 regles per confiança:")
        for i, rule in enumerate(summary["top_rules"][:3]):
            print(f"   {i+1}. {rule['rule_description']} (confiança: {rule.get('confidence', 0):.3f})")
    
    # Guardar resultats
    if config["output"]["save_results"]:
        inferer.save_rules(output_dirs["rules"] / "inferred_rules.json")
        
        # Crear informes detallats
        print("\n📋 Generant informes de regles...")
        rule_reports = []
        
        for rule in rules[:10]:  # Limitar a 10 per rendiment
            report = create_rule_report(rule, [phi_sequence])
            rule_reports.append(report)
        
        with open(output_dirs["reports"] / "rule_reports.json", 'w') as f:
            json.dump(rule_reports, f, indent=2)
        
        print(f"   Informes guardats per {len(rule_reports)} regles")
    
    return rules


def run_validation(rules: List[Dict[str, Any]], patterns: List[Dict[str, Any]],
                  additional_sequences: List[str], config: Dict[str, Any],
                  output_dirs: Dict[str, Path]) -> Dict[str, Any]:
    """Executa la validació de regles."""
    print("\n✅ FASE 4: Validació de Regles")
    print("=" * 60)
    
    validation_config = config["validation"]
    
    # Crear i configurar validador
    validator = Validator(
        validation_split=validation_config["validation_split"],
        min_validation_score=validation_config["min_validation_score"],
        stability_threshold=validation_config["stability_threshold"]
    )
    
    # Validar regles
    if validation_config.get("cross_validation_folds", 0) > 1:
        print("🔄 Executant validació creuada...")
        validation_results = validator.cross_validate_rules(
            rules, additional_sequences, 
            k_folds=validation_config["cross_validation_folds"]
        )
    else:
        validation_results = validator.validate_rules(rules, patterns, additional_sequences)
    
    print(f"✅ Validació completada")
    
    # Mostrar resum
    summary = validator.get_validation_summary()
    if summary.get("total_rules_tested", 0) > 0:
        print(f"   Regles testejades: {summary['total_rules_tested']}")
        print(f"   Precisió mitjana: {summary['mean_accuracy']:.3f}")
        print(f"   Estabilitat mitjana: {summary['mean_stability']:.3f}")
        print(f"   Regles sobre llindar: {summary['rules_above_threshold']}")
        
        # Mostrar millors regles
        if summary["best_performing_rules"]:
            print("\n📊 Top 3 regles per rendiment:")
            for i, (rule_id, perf) in enumerate(summary["best_performing_rules"][:3]):
                print(f"   {i+1}. {rule_id} - Precisió: {perf['accuracy']:.3f}")
    
    # Mostrar recomanacions
    recommendations = validation_results.get('recommendations', [])
    if recommendations:
        print("\n💡 Recomanacions del sistema:")
        for rec in recommendations[:5]:  # Mostrar només les primeres 5
            print(f"   • {rec}")
    
    # Guardar resultats
    if config["output"]["save_results"]:
        validator.save_validation_results(output_dirs["validation"] / "validation_results.json")
        print("   Resultats de validació guardats")
    
    return validation_results


def create_visualizations(phi_sequence: str, patterns: List[Dict[str, Any]],
                         rules: List[Dict[str, Any]], validation_results: Dict[str, Any],
                         output_dirs: Dict[str, Path]) -> None:
    """Crea visualitzacions dels resultats."""
    print("\n🎨 FASE 5: Generació de Visualitzacions")
    print("=" * 60)
    
    try:
        from utils.visualization import (
            plot_phi_sequence, plot_phi_hilbert, plot_pattern_distribution,
            plot_rule_performance, plot_validation_results
        )
        
        viz_dir = output_dirs["visualizations"]
        
        # 1. Seqüència Φ
        print("   Creant gràfic de seqüència Φ...")
        plot_phi_sequence(
            phi_sequence, 
            title="Seqüència Φ Generada",
            save_path=str(viz_dir / "phi_sequence.png")
        )
        
        # 2. Mapa de Hilbert
        print("   Creant mapa de Hilbert...")
        plot_phi_hilbert(
            phi_sequence,
            title="Seqüència Φ - Corba de Hilbert",
            save_path=str(viz_dir / "phi_hilbert.png")
        )
        
        # 3. Distribució de patrons
        if patterns:
            print("   Creant distribució de patrons...")
            plot_pattern_distribution(
                patterns,
                title="Distribució de Patrons Pₖ Detectats",
                save_path=str(viz_dir / "pattern_distribution.png")
            )
        
        # 4. Rendiment de regles
        if rules:
            print("   Creant gràfic de rendiment de regles...")
            plot_rule_performance(
                rules,
                title="Rendiment de Regles ωₖ Inferides",
                save_path=str(viz_dir / "rule_performance.png")
            )
        
        # 5. Resultats de validació
        if validation_results.get('rule_scores'):
            print("   Creant gràfic de validació...")
            plot_validation_results(
                validation_results,
                title="Resultats de Validació",
                save_path=str(viz_dir / "validation_results.png")
            )
        
        # 6. Dashboard resum
        print("   Creant dashboard resum...")
        create_summary_dashboard(
            phi_sequence, patterns, rules, validation_results,
            save_path=str(viz_dir / "summary_dashboard.png")
        )
        
        print("✅ Visualitzacions creades")
        
    except ImportError as e:
        print(f"⚠️  No es poden crear visualitzacions: {e}")
        print("   Instal·la matplotlib i seaborn per habilitar visualitzacions")


def generate_final_report(phi_sequence: str, patterns: List[Dict[str, Any]],
                         rules: List[Dict[str, Any]], validation_results: Dict[str, Any],
                         metadata: Dict[str, Any], output_dirs: Dict[str, Path]) -> None:
    """Genera un informe final complet."""
    print("\n📄 Generant informe final...")
    
    # Calcular estadístiques globals
    phi_length = len(phi_sequence)
    num_patterns = len(patterns)
    num_rules = len(rules)
    
    # Mètriques de qualitat
    if patterns:
        avg_pattern_length = sum(len(p['pattern_data']) for p in patterns) / len(patterns)
        avg_recurrence = sum(p.get('recurrence', 0) for p in patterns) / len(patterns)
    else:
        avg_pattern_length = 0
        avg_recurrence = 0
    
    if rules:
        avg_confidence = sum(r.get('confidence', 0) for r in rules) / len(rules)
        avg_precision = sum(r.get('precision', 0) for r in rules) / len(rules)
    else:
        avg_confidence = 0
        avg_precision = 0
    
    # Crear informe
    final_report = {
        "experiment_summary": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "phi_generation": {
                "sequence_length": phi_length,
                "iterations_completed": metadata.get('iterations_completed', 0),
                "collapse_rule": metadata.get('collapse_rule', 'unknown'),
                "convergence_reached": metadata.get('convergence_reached', False)
            },
            "pattern_detection": {
                "total_patterns": num_patterns,
                "average_pattern_length": avg_pattern_length,
                "average_recurrence": avg_recurrence
            },
            "rule_inference": {
                "total_rules": num_rules,
                "average_confidence": avg_confidence,
                "average_precision": avg_precision
            },
            "validation": validation_results.get('overall_metrics', {}),
            "recommendations": validation_results.get('recommendations', [])
        },
        "detailed_results": {
            "patterns": patterns[:20],  # Només els primers 20
            "rules": rules[:20],        # Només les primeres 20
            "validation_summary": validation_results
        }
    }
    
    # Guardar informe
    with open(output_dirs["reports"] / "final_report.json", 'w') as f:
        json.dump(final_report, f, indent=2)
    
    print(f"✅ Informe final guardat a {output_dirs['reports'] / 'final_report.json'}")
    
    # Mostrar resum a consola
    print("\n" + "="*80)
    print("📊 RESUM FINAL DE L'EXPERIMENT HSI")
    print("="*80)
    print(f"🔧 Seqüència Φ: {phi_length:,} bits generats")
    print(f"🔍 Patrons Pₖ: {num_patterns} detectats")
    print(f"🧠 Regles ωₖ: {num_rules} inferides")
    print(f"✅ Validació: {validation_results.get('overall_metrics', {}).get('mean_accuracy', 0):.3f} precisió mitjana")
    
    if metadata.get('convergence_reached'):
        print("🎯 Convergència cap a φ (1.618) assolida!")
    
    print("="*80)


def main():
    """Funció principal del sistema d'agents HSI."""
    parser = argparse.ArgumentParser(
        description="Sistema d'Agents per la Hipòtesi de Singularitat Informacional",
        epilog="💡 Edita config.json per personalitzar l'experiment"
    )
    parser.add_argument("--config", "-c", default="config.json",
                       help="Fitxer de configuració JSON (defecte: config.json)")
    parser.add_argument("--output", "-o", default="results", help="Directori de sortida")
    parser.add_argument("--no-viz", action="store_true", help="Desactivar visualitzacions")
    parser.add_argument("--quiet", "-q", action="store_true", help="Mode silenciós")

    args = parser.parse_args()

    # Configurar sortida
    if args.quiet:
        import sys
        sys.stdout = open('/dev/null', 'w') if hasattr(sys, 'stdout') else sys.stdout

    print("🚀 Sistema d'Agents HSI - Hipòtesi de Singularitat Informacional")
    print("=" * 80)
    print("Autor: Iban Borràs amb col·laboració d'Augment Agent")
    print("Data: Gener 2025")
    print("=" * 80)

    # Carregar configuració (sempre des de fitxer)
    config = load_config(args.config)
    
    # Configurar directoris de sortida
    output_dirs = setup_output_directories(args.output)
    
    try:
        # FASE 1: Generació Φ
        phi_sequence, snapshots, metadata, additional_sequences = run_phi_generation(
            config, output_dirs
        )
        
        # FASE 2: Detecció de patrons
        patterns = run_pattern_detection(phi_sequence, config, output_dirs)
        
        # FASE 3: Inferència de regles
        rules = run_rule_inference(patterns, phi_sequence, config, output_dirs)
        
        # FASE 4: Validació
        validation_results = run_validation(
            rules, patterns, additional_sequences, config, output_dirs
        )
        
        # FASE 5: Visualitzacions
        if config["output"]["create_visualizations"] and not args.no_viz:
            create_visualizations(
                phi_sequence, patterns, rules, validation_results, output_dirs
            )
        
        # Informe final
        generate_final_report(
            phi_sequence, patterns, rules, validation_results, metadata, output_dirs
        )
        
        print("\n🎉 Experiment completat amb èxit!")
        print(f"📁 Resultats disponibles a: {output_dirs['base']}")
        
    except KeyboardInterrupt:
        print("\n⚠️  Experiment interromput per l'usuari")
    except Exception as e:
        print(f"\n❌ Error durant l'experiment: {e}")
        raise


if __name__ == "__main__":
    main()
