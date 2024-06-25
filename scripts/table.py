import json
import numpy as np

def fill_latex_table(values):
    # Ensure the values list has the correct number of elements
    if len(values) != 80:
        raise ValueError("The values list must contain exactly 80 elements.")

    # Define the LaTeX table template with placeholders
    template = r"""
    % Please add the following required packages to your document preamble:
    % \usepackage{{booktabs}}
    % \usepackage{{multirow}}
    \begin{{table}}[]
    \begin{{tabular}}{{@{{}}lllllllll@{{}}}}
    \toprule
    \multicolumn{{1}}{{c|}}{{\multirow{{2}}{{*}}{{Algorithm}}}} & \multicolumn{{2}}{{c|}}{{Confidence}}                             & \multicolumn{{2}}{{c|}}{{Support}}                                & \multicolumn{{2}}{{c|}}{{Lift}}                                   & \multicolumn{{2}}{{c}}{{Coverage}}                               \\
    \multicolumn{{1}}{{c|}}{{}}                           & \multicolumn{{1}}{{c}}{{Grouping}} & \multicolumn{{1}}{{c|}}{{Default}} & \multicolumn{{1}}{{c}}{{Grouping}} & \multicolumn{{1}}{{c|}}{{Default}} & \multicolumn{{1}}{{c}}{{Grouping}} & \multicolumn{{1}}{{c|}}{{Default}} & \multicolumn{{1}}{{c}}{{Grouping}} & \multicolumn{{1}}{{c|}}{{Default}} \\ \midrule
    \multicolumn{{9}}{{c}}{{\textbf{{LBNL FDD}}}}                                                                                                                                                                                                                                                                  \\
    \multicolumn{{1}}{{l|}}{{DE}}                         & {0}                        & \multicolumn{{1}}{{l|}}{{{1}}}        & {2}                        & \multicolumn{{1}}{{l|}}{{{3}}}        & {4}                        & \multicolumn{{1}}{{l|}}{{{5}}}        & {6}                        & {7}                             \\
    \multicolumn{{1}}{{l|}}{{BAT}}                        & {8}                        & \multicolumn{{1}}{{l|}}{{{9}}}        & {10}                       & \multicolumn{{1}}{{l|}}{{{11}}}       & {12}                       & \multicolumn{{1}}{{l|}}{{{13}}}       & {14}                       & {15}                            \\
    \multicolumn{{1}}{{l|}}{{GWO}}                        & {16}                       & \multicolumn{{1}}{{l|}}{{{17}}}       & {18}                       & \multicolumn{{1}}{{l|}}{{{19}}}       & {20}                       & \multicolumn{{1}}{{l|}}{{{21}}}       & {22}                       & {23}                            \\
    \multicolumn{{1}}{{l|}}{{HHO}}                        & {24}                       & \multicolumn{{1}}{{l|}}{{{25}}}       & {26}                       & \multicolumn{{1}}{{l|}}{{{27}}}       & {28}                       & \multicolumn{{1}}{{l|}}{{{29}}}       & {30}                       & {31}                            \\
    \multicolumn{{1}}{{l|}}{{SCA}}                        & {32}                       & \multicolumn{{1}}{{l|}}{{{33}}}       & {34}                       & \multicolumn{{1}}{{l|}}{{{35}}}       & {36}                       & \multicolumn{{1}}{{l|}}{{{37}}}       & {38}                       & {39}                            \\
    \multicolumn{{9}}{{c}}{{\textbf{{LeakkDB}}}}                                                                                                                                                                                                                                                                   \\
    \multicolumn{{1}}{{l|}}{{DE}}                         & {40}                       & \multicolumn{{1}}{{l|}}{{{41}}}       & {42}                       & \multicolumn{{1}}{{l|}}{{{43}}}       & {44}                       & \multicolumn{{1}}{{l|}}{{{45}}}       & {46}                       & {47}                            \\
    \multicolumn{{1}}{{l|}}{{BAT}}                        & {48}                       & \multicolumn{{1}}{{l|}}{{{49}}}       & {50}                       & \multicolumn{{1}}{{l|}}{{{51}}}       & {52}                       & \multicolumn{{1}}{{l|}}{{{53}}}       & {54}                       & {55}                            \\
    \multicolumn{{1}}{{l|}}{{GWO}}                        & {56}                       & \multicolumn{{1}}{{l|}}{{{57}}}       & {58}                       & \multicolumn{{1}}{{l|}}{{{59}}}       & {60}                       & \multicolumn{{1}}{{l|}}{{{61}}}       & {62}                       & {63}                            \\
    \multicolumn{{1}}{{l|}}{{HHO}}                        & {64}                       & \multicolumn{{1}}{{l|}}{{{65}}}       & {66}                       & \multicolumn{{1}}{{l|}}{{{67}}}       & {68}                       & \multicolumn{{1}}{{l|}}{{{69}}}       & {70}                       & {71}                            \\
    \multicolumn{{1}}{{l|}}{{SCA}}                        & {72}                       & \multicolumn{{1}}{{l|}}{{{73}}}       & {74}                       & \multicolumn{{1}}{{l|}}{{{75}}}       & {76}                       & \multicolumn{{1}}{{l|}}{{{77}}}       & {78}                       & {79}
    \end{{tabular}}
    \end{{table}}
    """

    # Format the template with the provided values
    filled_table = template.format(*values)
    return filled_table

def fill_latex_table_v2(values):
    # Ensure the values list has the correct number of elements
    if len(values) != 60:
        raise ValueError("The values list must contain exactly 60 elements.")

    # Define the LaTeX table template with placeholders
    template = r"""
    % Please add the following required packages to your document preamble:
    % \usepackage{{multirow}}
    \begin{{table}}[]
    \begin{{tabular}}{{lllllll}}
    \hline
    \multicolumn{{1}}{{c|}}{{\multirow{{2}}{{*}}{{Algorithm}}}} & \multicolumn{{2}}{{c|}}{{Zhang's metric}}                         & \multicolumn{{2}}{{c|}}{{Yule's Q}}                               & \multicolumn{{2}}{{c}}{{Number of rules}}                        \\
    \multicolumn{{1}}{{c|}}{{}}                           & \multicolumn{{1}}{{c}}{{Grouping}} & \multicolumn{{1}}{{c|}}{{Default}} & \multicolumn{{1}}{{c}}{{Grouping}} & \multicolumn{{1}}{{c|}}{{Default}} & \multicolumn{{1}}{{c}}{{Grouping}} & \multicolumn{{1}}{{c}}{{Default}} \\ \hline
    \multicolumn{{7}}{{c}}{{\textbf{{LBNL FDD}}}}                                                                                                                                                                                                    \\
    \multicolumn{{1}}{{l|}}{{DE}}                         & {0}                              & \multicolumn{{1}}{{l|}}{{{1}}}        & {2}                              & \multicolumn{{1}}{{l|}}{{{3}}}        & {4}                              & {5}                             \\
    \multicolumn{{1}}{{l|}}{{BAT}}                        & {6}                              & \multicolumn{{1}}{{l|}}{{{7}}}        & {8}                              & \multicolumn{{1}}{{l|}}{{{9}}}        & {10}                             & {11}                            \\
    \multicolumn{{1}}{{l|}}{{GWO}}                        & {12}                             & \multicolumn{{1}}{{l|}}{{{13}}}       & {14}                             & \multicolumn{{1}}{{l|}}{{{15}}}       & {16}                             & {17}                            \\
    \multicolumn{{1}}{{l|}}{{HHO}}                        & {18}                             & \multicolumn{{1}}{{l|}}{{{19}}}       & {20}                             & \multicolumn{{1}}{{l|}}{{{21}}}       & {22}                             & {23}                            \\
    \multicolumn{{1}}{{l|}}{{SCA}}                        & {24}                             & \multicolumn{{1}}{{l|}}{{{25}}}       & {26}                             & \multicolumn{{1}}{{l|}}{{{27}}}       & {28}                             & {29}                            \\
    \multicolumn{{7}}{{c}}{{\textbf{{LeakkDB}}}}                                                                                                                                                                                                     \\
    \multicolumn{{1}}{{l|}}{{DE}}                         & {30}                             & \multicolumn{{1}}{{l|}}{{{31}}}       & {32}                             & \multicolumn{{1}}{{l|}}{{{33}}}       & {34}                             & {35}                            \\
    \multicolumn{{1}}{{l|}}{{BAT}}                        & {36}                             & \multicolumn{{1}}{{l|}}{{{37}}}       & {38}                             & \multicolumn{{1}}{{l|}}{{{39}}}       & {40}                             & {41}                            \\
    \multicolumn{{1}}{{l|}}{{GWO}}                        & {42}                             & \multicolumn{{1}}{{l|}}{{{43}}}       & {44}                             & \multicolumn{{1}}{{l|}}{{{45}}}       & {46}                             & {47}                            \\
    \multicolumn{{1}}{{l|}}{{HHO}}                        & {48}                             & \multicolumn{{1}}{{l|}}{{{49}}}       & {50}                             & \multicolumn{{1}}{{l|}}{{{51}}}       & {52}                             & {53}                            \\
    \multicolumn{{1}}{{l|}}{{SCA}}                        & {54}                             & \multicolumn{{1}}{{l|}}{{{55}}}       & {56}                             & \multicolumn{{1}}{{l|}}{{{57}}}       & {58}                             & {59}
    \end{{tabular}}
    \end{{table}}
    """

    # Format the template with the provided values
    filled_table = template.format(*values)
    return filled_table

def fill_latex_table_v3(values):
    # Ensure the values list has the correct number of elements
    if len(values) != 240:
        raise ValueError("The values list must contain exactly 240 elements.")

    # Define the LaTeX table template with placeholders
    template = r"""
    \begin{{table}}[h]
    \centering
    \setlength\tabcolsep{{3.5pt}} % default value: 6pt
    \begin{{tabular}}{{@{{}}lcccccccccc@{{}}}}
    \toprule
    \multicolumn{{1}}{{c}}{{\multirow{{2}}{{*}}{{Evaluations}}}} & \multicolumn{{2}}{{c}}{{DE}} & \multicolumn{{2}}{{c}}{{BAT}} & \multicolumn{{2}}{{c}}{{GWO}} & \multicolumn{{2}}{{c}}{{HHO}} & \multicolumn{{2}}{{c}}{{SCA}} \\
    \multicolumn{{1}}{{c}}{{}}                             & Group     & Default    & Group     & Default     & Group     & Default     & Group     & Default     & Group     & Default     \\ \midrule
                                                     & \multicolumn{{10}}{{c}}{{\textbf{{Confidence}}}}                                                                                       \\
    10k                                              & {0}      & {1}       & {2}      & {3}        & {4}      & {5}        & {6}      & {7}        & {8}      & {9}        \\
    25k                                              & {10}      & {11}       & {12}      & {13}        & {14}      & {15}        & {16}      & {17}        & {18}      & {19}        \\
    50k                                              & {20}      & {21}       & {22}      & {23}        & {24}      & {25}        & {26}      & {27}        & {28}      & {29}        \\
    100k                                             & {30}      & {31}       & {32}      & {33}        & {34}      & {35}        & {36}      & {37}        & {38}      & {39}        \\ \midrule
                                                     & \multicolumn{{10}}{{c}}{{\textbf{{Support}}}}                                                                                          \\
    10k                                              & {40}      & {41}       & {42}      & {43}        & {44}      & {45}        & {46}      & {47}        & {48}      & {49}        \\
    25k                                              & {50}      & {51}       & {52}      & {53}        & {54}      & {55}        & {56}      & {57}        & {58}      & {59}        \\
    50k                                              & {60}      & {61}       & {62}      & {63}        & {64}      & {65}        & {66}      & {67}        & {68}      & {69}        \\
    100k                                             & {70}      & {71}       & {72}      & {73}        & {74}      & {75}        & {76}      & {77}        & {78}      & {79}        \\ \midrule
                                                     & \multicolumn{{10}}{{c}}{{\textbf{{Lift}}}}                                                                                             \\
    10k                                              & {80}      & {81}       & {82}      & {83}        & {84}      & {85}        & {86}      & {87}        & {88}      & {89}        \\
    25k                                              & {90}      & {91}       & {92}      & {93}        & {94}      & {95}        & {96}      & {97}        & {98}      & {99}        \\
    50k                                              & {100}      & {101}       & {102}      & {103}        & {104}      & {105}        & {106}      & {107}        & {108}      & {109}        \\
    100k                                             & {110}      & {111}       & {112}      & {113}        & {114}      & {115}        & {116}      & {117}        & {118}      & {119}        \\ \midrule
                                                     & \multicolumn{{10}}{{c}}{{\textbf{{Coverage}}}}                                                                                         \\
    10k                                              & {120}      & {121}       & {122}      & {123}        & {124}      & {125}        & {126}      & {127}        & {128}      & {129}        \\
    25k                                              & {130}      & {131}       & {132}      & {133}        & {134}      & {135}        & {136}      & {137}        & {138}      & {139}        \\
    50k                                              & {140}      & {141}       & {142}      & {143}        & {144}      & {145}        & {146}      & {147}        & {148}      & {149}        \\
    100k                                             & {150}      & {151}       & {152}      & {153}        & {154}      & {155}        & {156}      & {157}        & {158}      & {159}        \\ \midrule
                                                     & \multicolumn{{10}}{{c}}{{\textbf{{Zhang's metric}}}}                                                                                   \\
    10k                                              & {160}      & {161}       & {162}      & {163}        & {164}      & {165}        & {166}      & {167}        & {168}      & {169}        \\
    25k                                              & {170}      & {171}       & {172}      & {173}        & {174}      & {175}        & {176}      & {177}        & {178}      & {179}        \\
    50k                                              & {180}      & {181}       & {182}      & {183}        & {184}      & {185}        & {186}      & {187}        & {188}      & {189}        \\
    100k                                             & {190}      & {191}       & {192}      & {193}        & {194}      & {195}        & {196}      & {197}        & {198}      & {199}        \\ \midrule
                                                     & \multicolumn{{10}}{{c}}{{\textbf{{Yule's Q}}}}                                                                                         \\
    10k                                              & {200}      & {201}       & {202}      & {203}        & {204}      & {205}        & {206}      & {207}        & {208}      & {209}        \\
    25k                                              & {210}      & {211}       & {212}      & {213}        & {214}      & {215}        & {216}      & {217}        & {218}      & {219}        \\
    50k                                              & {220}      & {221}       & {222}      & {223}        & {224}      & {225}        & {226}      & {227}        & {228}      & {229}        \\
    100k                                             & {230}      & {231}       & {232}      & {233}        & {234}      & {235}        & {236}      & {237}        & {238}      & {239}        \\ \midrule
    \end{{tabular}}
    \caption{{Comparison of methods for the different algorithms and various maximum evaluation settings}}
    \label{{table:evaluations}}
    \end{{table}}
    """

    # Format the template with the provided values
    filled_table = template.format(*values)
    return filled_table

algos = ['DE', 'BAT', 'GWO', 'HHO', 'SCA']
datasets = ['lbnl_fdd', 'leakdb']

results1 = []
results2 = []

for dataset in datasets:
    for algo in algos:
        path = f"../results/{algo}/{dataset}/results_50000.json"

        if dataset == "leakdb":
            path = f"../results/{algo}/{dataset}/results_50000_sf(0.2).json"

        with open(path, "r") as f:
            file_results = list(json.load(f))
            grouped = file_results[0]
            regular = file_results[1]

            mean_confidence_grouped = grouped["mean_confidence"]
            mean_confidence = regular["mean_confidence"]
            mean_support_grouped = grouped["mean_support"]
            mean_support = regular["mean_support"]
            mean_lift_grouped = grouped["mean_lift"]
            mean_lift = regular["mean_lift"]
            mean_coverage_grouped = grouped["mean_coverage"]
            mean_coverage = regular["mean_coverage"]
            mean_zhang_grouped = grouped["mean_zhang"]
            mean_zhang = regular["mean_zhang"]
            mean_yulesQ_grouped = grouped["mean_yulesQ"]
            mean_yulesQ = regular["mean_yulesQ"]
            mean_n_rules_learned_grouped = grouped["mean_n_rules_learned"]
            mean_n_rules_learned = regular["mean_n_rules_learned"]

            results1.extend([
                mean_confidence_grouped, mean_confidence,
                mean_support_grouped, mean_support,
                mean_lift_grouped, mean_lift,
                mean_coverage_grouped, mean_coverage,
            ])

            results2.extend([
                mean_zhang_grouped, mean_zhang,
                mean_yulesQ_grouped, mean_yulesQ,
                mean_n_rules_learned_grouped, mean_n_rules_learned,
            ])

results3 = [i for i in range(240, 480)]

# Example usage
rounded_results1 = [round(result, 3) for result in results1]
rounded_results2 = [round(result, 3) for result in results2]
rounded_results3 = [round(result, 3) for result in results3]

latex_code = fill_latex_table(rounded_results1)
# print(latex_code)

latex_code_v2 = fill_latex_table_v2(rounded_results2)
# print(latex_code_v2)

latex_code_v3 = fill_latex_table_v3(rounded_results3)
# print(latex_code_v3)