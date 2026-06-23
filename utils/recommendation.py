import streamlit as st

def generate_recommendations(top_explanations):
    """
    Analyzes the top SHAP features contributing positively to PMOS risk
    and yields targeted medical, dietary, and lifestyle recommendations.
    No emojis are used.
    """
    recommendations = []
    
    # Filter for features that are contributing to INCREASING the risk (shap > 0)
    risk_drivers = [x for x in top_explanations if x['shap'] > 0]
    
    # Track which recommendations have been added to prevent duplicates
    added_recommendations = set()
    
    for item in risk_drivers:
        feature_name = item['feature']
        
        # Limit to top 5 risk drivers to keep suggestions highly focused and personalized
        if len(recommendations) >= 5:
            break
            
        if feature_name == 'amh_ng_ml' and 'amh' not in added_recommendations:
            recommendations.append({
                'title': "Pengelolaan Kadar Hormon AMH Tinggi",
                'type': 'warning',
                'description': (
                    "Kadar AMH (Anti-Müllerian Hormone) yang tinggi merupakan indikator kuat adanya banyak folikel kecil (polikistik) di ovarium. "
                    "Disarankan untuk berkonsultasi dengan Dokter Spesialis Obstetri dan Ginekologi (Sp.OG) atau Spesialis Endokrinologi Reproduksi "
                    "untuk evaluasi hormonal menyeluruh dan pemantauan ovulasi."
                ),
                'icon': ""
            })
            added_recommendations.add('amh')
            
        elif feature_name in ['bmi', 'weight_gain_y_n'] and 'weight' not in added_recommendations:
            recommendations.append({
                'title': "Manajemen Berat Badan dan Sensitivitas Insulin",
                'type': 'warning',
                'description': (
                    "Peningkatan Indeks Massa Tubuh (BMI) atau berat badan yang signifikan berkolaborasi erat dengan resistensi insulin pada PMOS. "
                    "Terapkan diet rendah indeks glikemik (Low GI), batasi konsumsi gula sederhana, dan fokuslah pada makanan padat nutrisi. "
                    "Penurunan berat badan 5-10% terbukti dapat membantu memulihkan siklus menstruasi dan meningkatkan kesuburan."
                ),
                'icon': ""
            })
            added_recommendations.add('weight')
            
        elif feature_name in ['cycle_r_i', 'cycle_length_days'] and 'cycle' not in added_recommendations:
            recommendations.append({
                'title': "Evaluasi Siklus Menstruasi yang Tidak Teratur",
                'type': 'warning',
                'description': (
                    "Siklus menstruasi yang memanjang (>35 hari) atau tidak teratur merupakan tanda anovulasi (sel telur tidak matang). "
                    "Catat siklus menstruasi secara detail dan konsultasikan opsi regulasi siklus (seperti terapi hormonal atau modifikasi gaya hidup) "
                    "bersama dokter spesialis kandungan Anda."
                ),
                'icon': ""
            })
            added_recommendations.add('cycle')
            
        elif feature_name in ['lh_miu_ml', 'fsh_lh'] and 'hormone_ratio' not in added_recommendations:
            recommendations.append({
                'title': "Regulasi Rasio Hormon LH/FSH",
                'type': 'warning',
                'description': (
                    "Rasio LH terhadap FSH yang terbalik (LH lebih tinggi dari FSH) mengganggu pematangan sel telur. "
                    "Lakukan manajemen stres yang baik (yoga, meditasi) karena hormon stres (kortisol) dapat mengacaukan regulasi LH. "
                    "Pastikan tidur yang cukup (7-8 jam per malam) untuk mengoptimalkan sekresi hormon reproduksi."
                ),
                'icon': ""
            })
            added_recommendations.add('hormone_ratio')
            
        elif feature_name in ['follicle_no_l', 'follicle_no_r'] and 'follicle' not in added_recommendations:
            recommendations.append({
                'title': "Pemantauan Gambaran Polikistik Ovarium",
                'type': 'warning',
                'description': (
                    "Jumlah folikel ovarium yang tinggi (>12 folikel pada satu ovarium) merupakan ciri morfologi polikistik. "
                    "Gambaran USG ini memerlukan korelasi klinis dengan gejala lain (seperti gangguan menstruasi atau tanda androgenik). "
                    "Lakukan pemeriksaan USG transvaginal berkala sesuai saran dokter spesialis kandungan."
                ),
                'icon': ""
            })
            added_recommendations.add('follicle')
            
        elif feature_name in ['hair_growth_y_n', 'skin_darkening_y_n', 'pimples_y_n', 'hair_loss_y_n'] and 'androgen' not in added_recommendations:
            recommendations.append({
                'title': "Penanganan Gejala Kelebihan Androgen (Hiperandrogenisme)",
                'type': 'warning',
                'description': (
                    "Gejala seperti jerawat parah, pertumbuhan bulu berlebih (hirsutisme), penggelapan kulit (acanthosis nigricans), atau kerontokan rambut "
                    "menunjukkan aktivitas hormon androgen (hormon maskulin) yang berlebih. "
                    "Dokter dapat meresepkan terapi anti-androgen atau menyarankan perawatan topikal yang sesuai dengan kondisi kulit Anda."
                ),
                'icon': ""
            })
            added_recommendations.add('androgen')
            
        elif feature_name == 'fast_food_y_n' and 'fast_food' not in added_recommendations:
            recommendations.append({
                'title': "Pembatasan Makanan Cepat Saji (Fast Food)",
                'type': 'info',
                'description': (
                    "Konsumsi fast food secara rutin memicu peradangan sistemik dan resistensi insulin, memperburuk gejala PMOS. "
                    "Kurangi konsumsi fast food secara bertahap dan gantilah dengan masakan rumah yang diolah minimal (whole foods) "
                    "yang kaya serat, lemak sehat (seperti alpukat, minyak zaitun), dan protein rendah lemak."
                ),
                'icon': ""
            })
            added_recommendations.add('fast_food')
            
        elif feature_name == 'reg_exercise_y_n' and 'exercise' not in added_recommendations:
            recommendations.append({
                'title': "Peningkatan Rutinitas Aktivitas Fisik",
                'type': 'info',
                'description': (
                    "Kurangnya olahraga rutin mengurangi sensitivitas sel terhadap insulin. "
                    "Mulai rutin berolahraga minimal 150 menit per minggu, kombinasi antara latihan kardio (jalan cepat, bersepeda) "
                    "dan latihan beban ringan. Olahraga sangat efektif mengontrol kadar gula darah meski tanpa penurunan berat badan."
                ),
                'icon': ""
            })
            added_recommendations.add('exercise')
            
        elif feature_name in ['bp_systolic_mmhg', 'bp_diastolic_mmhg', 'rbs_mg_dl'] and 'metabolic' not in added_recommendations:
            recommendations.append({
                'title': "Skrining Risiko Metabolik",
                'type': 'warning',
                'description': (
                    "Tekanan darah atau kadar gula darah sewaktu (RBS) yang tinggi menunjukkan adanya risiko sindrom metabolik yang sering menyertai PMOS. "
                    "Disarankan melakukan cek profil lipid (kolesterol), HbA1c, serta memantau tekanan darah secara berkala untuk mencegah "
                    "komplikasi jangka panjang seperti Diabetes Tipe 2 atau penyakit kardiovaskular."
                ),
                'icon': ""
            })
            added_recommendations.add('metabolic')

    # Fallback default recommendations if no risk drivers were found or list is empty
    if not recommendations:
        recommendations.append({
            'title': "Pola Hidup Sehat untuk Menjaga Kesehatan Reproduksi",
            'type': 'success',
            'description': (
                "Sistem tidak mendeteksi faktor risiko dominan yang mengkhawatirkan. Tetap pertahankan gaya hidup sehat dengan "
                "mengonsumsi makanan bergizi seimbang, rutin berolahraga, mengelola tingkat stres dengan baik, dan menjaga "
                "pola tidur teratur untuk terus mendukung kesehatan ovarium Anda."
            ),
            'icon': ""
        })
        recommendations.append({
            'title': "Pemeriksaan Kesehatan Rutin",
            'type': 'info',
            'description': (
                "Lakukan pemeriksaan ginekologis secara berkala sebagai tindakan preventif untuk memantau kesehatan sistem reproduksi Anda."
            ),
            'icon': ""
        })
        
    return recommendations
