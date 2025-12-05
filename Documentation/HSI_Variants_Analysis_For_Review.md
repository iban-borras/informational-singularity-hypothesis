# Anàlisi de Variants HSI Level 0 — Document per a Revisió Externa

**Autors:** Iban Barroso, amb assistència de Sophia (AI)
**Revisora:** Ariadna (anàlisi filosòfica i estructural)
**Data:** 2025-11-28
**Versió:** v33 (HSI v32 Aligned)
**Propòsit:** Document de guia per a revisió externa i preparació d'un apèndix formal al paper

---

## 0. Resum Executiu (Conclusions d'Ariadna)

**Veredicte filosòfic:**

> "Si hem d'escollir una variant com a representant canònic de la HSI al Nivell 0, ha de ser la **Variant B**. És la que millor compleix: AND global (almenys en esperit), simetria 01/10, 0 absorbent, estratificació inside→out, acumulació de microestats + col·lapse final únic."

> "La Variant D ja està jugant en un terreny **Nivell 1/Nivell 2**: és com si estigueres injectant una asimetria que hauria de ser emergent, no primordial."

**Criteris de fidelitat ontològica establerts:**
1. Col·lapse ideal = AND global
2. Simetria 01/10 al Nivell 0
3. 0 com a estat absorbent
4. Estratificació inside→out + acumulació de microestats
5. Unicitat del valor final observable

---

## 1. Context: L'Operador `simplifica` al Paper

Al Nivell 0 del paper (secció 7 i Formalització Matemàtica), es defineix l'operador fonamental:

$$R_{\alpha} = \text{simplifica}(0, R_{\alpha-1}, 1)$$

On:
- `0` representa el No-Res pur
- `R_{α-1}` és la història acumulada de tots els estats previs
- `1` representa el No-Res Absolut emergent
- `simplifica` és l'operador de col·lapse coherent

### 1.1 Definició Ontològica (del paper)

> "L'operador `simplifica` representa el col·lapse coherent de la tensió entre el No-Res pur (0) i el No-Res Absolut (1). A nivell lògic, aquest col·lapse s'identifica de manera natural amb la conjunció booleana (AND)."

La taula de veritat AND:
```
0 ∧ 0 = 0
0 ∧ 1 = 0
1 ∧ 0 = 0
1 ∧ 1 = 1
```

### 1.2 Col·lapse Ideal vs. Discretitzacions Algorísmiques

El paper distingeix entre:

1. **Col·lapse ideal** (AND global):
   $$\mathrm{val}(S_\alpha) = \bigwedge_{i=0}^{n} s_i$$
   
   Com que la seqüència sempre comença amb 0, el valor col·lapsat ideal és sempre 0.

2. **Discretitzacions algorísmiques**: variants operatives que implementen la intuïció de l'AND mitjançant regles de reescriptura locals.

> "Des del punt de vista de la HSI, aquestes variants no representen versions contradictòries del principi fonamental, sinó **discretitzacions alternatives** d'una mateixa idea ontològica."

---

## 2. El Framework Comú de Totes les Variants

Totes les variants operen sobre el mateix marc:

```
Per cada iteració α ≥ 1:
1. Construir decay frame: Dα = (Accα)ABS
   - Accα = acumulació de tots els Φ previs
   - ABS = token Absolut ("1", "10" o "01")
   
2. Aplicar col·lapse estratificat inside→out:
   - Processar parèntesis més interiors primer
   - Acumular estats intermedis (ontològicament reals)
   
3. Obtenir observable Φα
```

### 2.1 Simplificació Base (`_simplify_base`)

Implementació comuna per defecte:

```python
def _simplify_base(seq: str) -> str:
    """Basal AND-rule simplification: 01 and 10 annihilate; compress runs."""
    # Aniquilar parells oposats (tensió 0-1)
    result = re.sub(r'(01|10)', '', seq)
    # Comprimir repeticions
    result = re.sub(r'0+', '0', result)
    result = re.sub(r'1+', '1', result)
    return result
```

**Interpretació ontològica:**
- `01 → ∅` i `10 → ∅`: La tensió entre No-Res i Absolut s'aniquila
- `00...0 → 0` i `11...1 → 1`: Estats idèntics es col·lapsen

---

## 3. Variant B — Gold Standard ⭐

### 3.1 Per què és el Gold Standard?

La Variant B és la implementació que millor captura la semàntica del paper:

1. **Respecta l'ordre estratificat**: Col·lapsa des de dins cap a fora
2. **Acumula micro-estats**: Cada estat intermedi és ontològicament real
3. **Produeix observable net**: Aplica simplificació final global

### 3.2 Algorisme Exacte

```python
# Variant B: Stratified inside→out + global finalize
elif variant == "B":
    while state != previous:
        previous = state
        # Acumular estat intermedi (HSI semantics!)
        accumulation += state
        # Col·lapsar només parèntesis més interiors
        next_state = _collapse_inside_parentheses_local(state)
        if len(next_state) == 1:
            state = next_state
            break
        state = next_state
    # Simplificació global final per garantir L0 observable
    if len(state) > 1:
        state = _collapse_global_ignore_parentheses(state)
```

### 3.3 Funció `_collapse_inside_parentheses_local`

```python
def _collapse_inside_parentheses_local(state: str, simplify_fn=None) -> str:
    """Simplifica NOMÉS continguts de parèntesis interiors.
    Preserva estructura externa per a passades estratificades."""
    sf = simplify_fn or _simplify_base
    def _collapse_match(match):
        inner_clean = _clean_sequence(match.group(0))
        return sf(inner_clean)
    return re.sub(r'\([01]+\)', _collapse_match, state)
```

### 3.4 Relació amb `simplifica` del Paper

| Aspecte del Paper | Implementació Variant B |
|-------------------|-------------------------|
| "col·lapse coherent" | `_simplify_base` amb AND |
| "estratificat" | `inside→out` via regex |
| "encapsulació" | Parèntesis `(Acc)1` |
| "història acumulada" | `accumulation += state` |

---

## 4. Variant D — Asimetria Mínima

### 4.1 Hipòtesi Científica que Prova

> "L'emergència d'ordre és robusta a la ruptura de simetria?"

La Variant D introdueix una asimetria direccional mínima per testar si l'ordre emergent depèn de la simetria perfecta 0↔1.

### 4.2 Funció de Simplificació Asimètrica

```python
def _simplify_variant_d(seq: str) -> str:
    """Minimal asymmetry: 10→∅, 01→0; then compress runs."""
    result = re.sub(r'10', '', seq)      # 10 s'aniquila
    result = re.sub(r'01', '0', result)  # 01 deixa residu 0
    result = re.sub(r'0+', '0', result)  # comprimir 0s
    result = re.sub(r'1+', '1', result)  # comprimir 1s
    return result
```

### 4.3 Diferència Clau amb Variant B

| Aspecte | Variant B | Variant D |
|---------|-----------|-----------|
| `10` | `→ ∅` (aniquilació) | `→ ∅` (aniquilació) |
| `01` | `→ ∅` (aniquilació) | `→ 0` (residu!) |
| Simetria | Perfecta | Trencada |

**Interpretació ontològica de l'asimetria:**
- A la Variant B, tant `01` com `10` representen tensions que s'aniquilen completament
- A la Variant D, `01` (No-Res seguit d'Absolut) deixa un residu de No-Res
- Això modela una "direccionalitat" en la degradació: l'Absolut que ve després del No-Res té un efecte diferent

### 4.4 Algorisme Variant D

```python
elif variant == "D":
    while state != previous:
        previous = state
        accumulation += state  # Acumula micro-estats
        next_state = _collapse_inside_parentheses(state,
                        simplify_fn=_simplify_variant_d)
        if len(next_state) == 1:
            state = next_state
            break
        state = next_state
    # Nota: NO aplica simplificació global final (diferència amb B)
```

---

## 5. Comparativa de Totes les Variants Actives

### 5.1 Taula Resum

| Variant | Simplificació | Final Global | Hipòtesi que Prova |
|---------|--------------|--------------|-------------------|
| **B** ⭐ | Base (simètrica) | Sí | Gold standard HSI |
| **D** | Asimètrica (01→0) | No | Robustesa a asimetria |
| **E** | Ordenada (01→∅, després 10→∅) | No | Efecte de l'ordre de passes |
| **F** | Base | Sí (una sola) | Híbrid B/sense-final |
| **G** | Base | No | Estructura crua preservada |
| **H** | Base + global per tick | Sí (cada pas) | Feedback continu |

### 5.2 Variants Eliminades (v33)

| Variant | Raó d'Eliminació |
|---------|-----------------|
| **A** | Incompatible amb principi d'ordre estratificat |
| **C** | Incompatible amb realitat de micro-estats |

---

## 6. Connexió amb el Paper: Per què Explorar Variants?

### 6.1 Justificació Teòrica (del paper, secció Formalització)

> "Des del punt de vista de la HSI, no disposem encara d'un criteri formal únic que seleccione una sola definició canònica de `simplifica`."

> "La recerca experimental explora diverses variants de l'operador `simplifica` i en compara el comportament (estabilitat, riquesa dels patrons emergents, estructura fractal, etc.). Aquesta estratègia permet:
> 1. Avaluar la robustesa de la HSI respecte de les decisions de discretització
> 2. Identificar quins esquemes són més compatibles amb la intuïció d'ordre coherent"

### 6.2 Pregunta Científica Central i Nota sobre φ

**Quina discretització de `simplifica` produeix estructures coherents amb el principi ontològic?**

Les propietats a estudiar inclouen:
- Estabilitat del creixement exponencial
- Riquesa dels patrons emergents
- Estructura fractal i autosimilitud

**Nota sobre φ (ràtio àuria):** Per coherència amb la maduresa actual de la HSI, la convergència cap a φ s'entén com una **hipòtesi exploratòria**, no com el centre de la validació. Entre les diferents mesures possibles de l'estructura emergent, analitzar si certes magnituds (com la dimensió fractal efectiva o ratios d'escala) s'acosten a valors especials com φ és una línia interessant, però no un criteri central de validació. (Suggeriment d'Ariadna)

---

## 7. Resultats Preliminars (Variant B, 23 iteracions)

### 7.1 Dades de l'Experiment

- **Iteracions completades:** 23
- **Mida de l'acumulació final:** ~213 GB
- **Creixement:** Exponencial confirmat
- **Temps total:** ~16 hores

### 7.2 Pendent d'Anàlisi

- Dimensió fractal (D_box)
- Convergència cap a φ
- Estructura espectral (FFT)
- Patrons Hilbert

---

## 8. Proposta d'Apèndix Formal per al Paper

### 8.1 Estructura Suggerida

**Apèndix A: Variants Algorísmiques de l'Operador Simplifica**

1. **A.1 Introducció**
   - Justificació de l'exploració de variants
   - Criteri de selecció (compatibilitat amb HSI v32)

2. **A.2 Framework Comú**
   - Decay frame: `(Accα)ABS`
   - Col·lapse estratificat inside→out
   - Acumulació de micro-estats

3. **A.3 Variant B (Gold Standard)**
   - Algorisme complet
   - Propietats matemàtiques
   - Resultats experimentals

4. **A.4 Variants Alternatives**
   - D: Asimetria mínima
   - E: Ordre de passes
   - F, G, H: Variacions de finalització

5. **A.5 Resultats Comparatius**
   - Creixement exponencial
   - Convergència a φ
   - Estructura fractal

6. **A.6 Conclusions**
   - Robustesa de la HSI
   - Variant òptima identificada

---

## 9. Preguntes per a la Revisora (Ariadna)

1. **Claredat ontològica:** L'explicació de per què cada variant és una "discretització" del mateix principi és prou clara?

2. **Rigor matemàtic:** Cal afegir lemes o proves formals sobre la confluència de les regles de reescriptura?

3. **Equilibri del paper:** L'apèndix hauria de ser tècnic-detallat o conceptual-accessible?

4. **Visualitzacions:** Quines gràfiques serien més il·lustratives per a l'apèndix?

5. **Nomenclatura:** Els noms de les variants (B, D, E...) són adequats o caldria noms més descriptius?

---

## 10. Referències Internes

- `level0/generator.py`: Implementació de totes les variants
- `Documentation/variants_spec.md`: Especificació tècnica detallada
- `Paper_Latex/main_cat.tex`: Paper principal (seccions 7, Formalització)
- `results/visualizations/`: Gràfiques generades

