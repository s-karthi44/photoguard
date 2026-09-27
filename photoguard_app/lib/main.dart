import 'dart:convert';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:image_picker/image_picker.dart';
import 'package:http/http.dart' as http;
import 'package:animate_do/animate_do.dart';

void main() {
  runApp(const PhotoGuardApp());
}

class PhotoGuardApp extends StatelessWidget {
  const PhotoGuardApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'PhotoGuard AI',
      debugShowCheckedModeBanner: false,
      theme: ThemeData.dark().copyWith(
        scaffoldBackgroundColor: const Color(0xFF0B0F19),
        colorScheme: const ColorScheme.dark(
          primary: Color(0xFF6366F1),
          secondary: Color(0xFF10B981),
          error: Color(0xFFEF4444),
          surface: Color(0xFF161E2E),
        ),
        textTheme: GoogleFonts.outfitTextTheme(ThemeData.dark().textTheme),
      ),
      home: const HomeScreen(),
    );
  }
}

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  Uint8List? _selectedImageBytes;
  String? _selectedImageName;
  bool _isLoading = false;
  Map<String, dynamic>? _predictionResult;
  Uint8List? _gradcamBytes;
  Uint8List? _elaBytes;
  Uint8List? _fftBytes;
  Uint8List? _splicingBytes;
  int _activeViewTab = 0; // 0 = Original, 1 = Grad-CAM, 2 = ELA, 3 = FFT, 4 = Splicing
  final ImagePicker _picker = ImagePicker();

  // Change this to your server IP if testing on a physical mobile device (e.g. http://192.168.1.X:8000)
  final String _apiUrl = "http://127.0.0.1:8000/api/v1/predict";

  Future<void> _pickImage(ImageSource source) async {
    final XFile? pickedFile = await _picker.pickImage(source: source);
    if (pickedFile != null) {
      final bytes = await pickedFile.readAsBytes();
      setState(() {
        _selectedImageBytes = bytes;
        _selectedImageName = pickedFile.name;
        _predictionResult = null;
        _gradcamBytes = null;
        _elaBytes = null;
        _fftBytes = null;
        _splicingBytes = null;
        _activeViewTab = 0;
      });
      _analyzePhoto();
    }
  }

  Future<void> _analyzePhoto() async {
    if (_selectedImageBytes == null) return;

    setState(() {
      _isLoading = true;
    });

    try {
      var request = http.MultipartRequest('POST', Uri.parse(_apiUrl));
      request.files.add(
        http.MultipartFile.fromBytes(
          'file',
          _selectedImageBytes!,
          filename: _selectedImageName ?? 'uploaded_photo.jpg',
        ),
      );

      var streamedResponse = await request.send();
      var response = await http.Response.fromStream(streamedResponse);

      if (response.statusCode == 200) {
        var data = jsonDecode(response.body);
        setState(() {
          _predictionResult = data;
          if (data['gradcam_base64'] != null) {
            _gradcamBytes = base64Decode(data['gradcam_base64']);
          }
          if (data['ela_base64'] != null) {
            _elaBytes = base64Decode(data['ela_base64']);
          }
          if (data['fft_base64'] != null) {
            _fftBytes = base64Decode(data['fft_base64']);
          }
          if (data['splicing_overlay_base64'] != null) {
            _splicingBytes = base64Decode(data['splicing_overlay_base64']);
          }
        });
      } else {
        _showErrorSnackBar("Server Error: ${response.statusCode}");
      }
    } catch (e) {
      _showErrorSnackBar("Connection Failed. Ensure FastAPI is running on port 8000.");
    } finally {
      setState(() {
        _isLoading = false;
      });
    }
  }

  void _showErrorSnackBar(String message) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text(message, style: const TextStyle(fontWeight: FontWeight.bold)),
        backgroundColor: const Color(0xFFEF4444),
        behavior: SnackBarBehavior.floating,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    String prediction = _predictionResult?['prediction']?.toString().toLowerCase() ?? '';
    bool isAi = prediction.contains('ai') || prediction.contains('spliced') || prediction.contains('fake');
    bool isUncertain = prediction.contains('uncertain');
    ThemeData theme = Theme.of(context);

    return Scaffold(
      appBar: AppBar(
        backgroundColor: const Color(0xFF0F172A),
        elevation: 0,
        centerTitle: true,
        title: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            const Icon(Icons.shield_outlined, color: Color(0xFF6366F1), size: 28),
            const SizedBox(width: 8),
            Text(
              "PhotoGuard AI",
              style: GoogleFonts.outfit(
                fontWeight: FontWeight.bold,
                fontSize: 24,
                letterSpacing: 0.5,
              ),
            ),
          ],
        ),
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // Image Preview Tabs
            if (_selectedImageBytes != null && _predictionResult != null)
              _buildViewTabs(),

            // Image Preview Container
            _buildImageContainer(isAi, isUncertain),

            const SizedBox(height: 20),

            // Action Buttons
            _buildActionButtons(),

            const SizedBox(height: 24),

            // Analysis Loading Overlay / Result Display
            if (_isLoading)
              _buildLoadingCard()
            else if (_predictionResult != null) ...[
              FadeInUp(
                duration: const Duration(milliseconds: 400),
                child: _buildVerdictCard(isAi, isUncertain, theme),
              ),
              const SizedBox(height: 16),
              FadeInUp(
                duration: const Duration(milliseconds: 500),
                child: _buildBreakdownCard(theme),
              ),
              const SizedBox(height: 16),
              FadeInUp(
                duration: const Duration(milliseconds: 600),
                child: _buildMetadataCard(theme),
              ),
            ],
          ],
        ),
      ),
    );
  }

  Widget _buildViewTabs() {
    return Container(
      margin: const EdgeInsets.only(bottom: 12),
      padding: const EdgeInsets.all(4),
      decoration: BoxDecoration(
        color: const Color(0xFF161E2E),
        borderRadius: BorderRadius.circular(12),
      ),
      child: SingleChildScrollView(
        scrollDirection: Axis.horizontal,
        child: Row(
          children: [
            _buildTabButton(0, "Original", Icons.photo_outlined),
            _buildTabButton(1, "Grad-CAM", Icons.insights_outlined),
            _buildTabButton(2, "ELA", Icons.grain_outlined),
            _buildTabButton(3, "FFT", Icons.graphic_eq_rounded),
            _buildTabButton(4, "Splicing", Icons.crop_free_rounded),
          ],
        ),
      ),
    );
  }

  Widget _buildTabButton(int index, String label, IconData icon) {
    bool isSelected = _activeViewTab == index;
    bool isAvailable = index == 0 ||
        (index == 1 && _gradcamBytes != null) ||
        (index == 2 && _elaBytes != null) ||
        (index == 3 && _fftBytes != null) ||
        (index == 4 && _splicingBytes != null);

    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 2),
      child: GestureDetector(
        onTap: isAvailable ? () => setState(() => _activeViewTab = index) : null,
        child: Container(
          padding: const EdgeInsets.symmetric(vertical: 10, horizontal: 12),
          decoration: BoxDecoration(
            color: isSelected ? const Color(0xFF6366F1) : Colors.transparent,
            borderRadius: BorderRadius.circular(8),
          ),
          child: Opacity(
            opacity: isAvailable ? 1.0 : 0.4,
            child: Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                Icon(icon, size: 18, color: isSelected ? Colors.white : Colors.grey),
                const SizedBox(width: 6),
                Text(
                  label,
                  style: TextStyle(
                    fontSize: 13,
                    fontWeight: FontWeight.bold,
                    color: isSelected ? Colors.white : Colors.grey,
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildImageContainer(bool isAi, bool isUncertain) {
    Widget? activeImage;
    if (_activeViewTab == 0 || _selectedImageBytes == null) {
      activeImage = _selectedImageBytes != null
          ? Image.memory(_selectedImageBytes!, fit: BoxFit.cover)
          : null;
    } else if (_activeViewTab == 1 && _gradcamBytes != null) {
      activeImage = Image.memory(_gradcamBytes!, fit: BoxFit.cover);
    } else if (_activeViewTab == 2 && _elaBytes != null) {
      activeImage = Image.memory(_elaBytes!, fit: BoxFit.cover);
    } else if (_activeViewTab == 3 && _fftBytes != null) {
      activeImage = Image.memory(_fftBytes!, fit: BoxFit.cover);
    } else if (_activeViewTab == 4 && _splicingBytes != null) {
      activeImage = Image.memory(_splicingBytes!, fit: BoxFit.cover);
    }

    Color borderColor;
    if (_predictionResult == null) {
      borderColor = Colors.white12;
    } else if (isAi) {
      borderColor = const Color(0xFFEF4444);
    } else if (isUncertain) {
      borderColor = const Color(0xFFF59E0B);
    } else {
      borderColor = const Color(0xFF10B981);
    }

    return Container(
      height: 340,
      decoration: BoxDecoration(
        color: const Color(0xFF161E2E),
        borderRadius: BorderRadius.circular(24),
        border: Border.all(
          color: borderColor,
          width: 2,
        ),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.4),
            blurRadius: 20,
            offset: const Offset(0, 8),
          )
        ],
      ),
      child: ClipRRect(
        borderRadius: BorderRadius.circular(22),
        child: activeImage == null
            ? Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Icon(
                    Icons.add_photo_alternate_outlined,
                    size: 72,
                    color: Colors.grey.shade700,
                  ),
                  const SizedBox(height: 16),
                  Text(
                    "Upload or capture a photo for verification",
                    style: TextStyle(
                      color: Colors.grey.shade500,
                      fontSize: 15,
                    ),
                  ),
                ],
              )
            : Stack(
                fit: StackFit.expand,
                children: [
                  activeImage,
                  if (_isLoading)
                    Container(
                      color: Colors.black.withOpacity(0.7),
                      child: const Center(
                        child: Column(
                          mainAxisSize: MainAxisSize.min,
                          children: [
                            CircularProgressIndicator(color: Color(0xFF6366F1)),
                            SizedBox(height: 20),
                            Text(
                              "Extracting Noise & Spectral Footprints...",
                              style: TextStyle(
                                color: Colors.white,
                                fontWeight: FontWeight.bold,
                              ),
                            )
                          ],
                        ),
                      ),
                    ),
                ],
              ),
      ),
    );
  }

  Widget _buildActionButtons() {
    return Row(
      children: [
        Expanded(
          child: _buildButton(
            onPressed: () => _pickImage(ImageSource.gallery),
            label: "Open Gallery",
            icon: Icons.photo_library_outlined,
            color: const Color(0xFF6366F1),
          ),
        ),
        const SizedBox(width: 16),
        Expanded(
          child: _buildButton(
            onPressed: () => _pickImage(ImageSource.camera),
            label: "Use Camera",
            icon: Icons.camera_alt_outlined,
            color: const Color(0xFF1E293B),
          ),
        ),
      ],
    );
  }

  Widget _buildButton({
    required VoidCallback onPressed,
    required String label,
    required IconData icon,
    required Color color,
  }) {
    return SizedBox(
      height: 52,
      child: ElevatedButton.icon(
        onPressed: onPressed,
        icon: Icon(icon, size: 20),
        label: Text(label, style: const TextStyle(fontWeight: FontWeight.bold)),
        style: ElevatedButton.styleFrom(
          backgroundColor: color,
          foregroundColor: Colors.white,
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
          elevation: 0,
        ),
      ),
    );
  }

  Widget _buildLoadingCard() {
    return Container(
      padding: const EdgeInsets.all(24),
      decoration: BoxDecoration(
        color: const Color(0xFF161E2E),
        borderRadius: BorderRadius.circular(20),
      ),
      child: const Row(
        children: [
          CircularProgressIndicator(color: Color(0xFF6366F1)),
          SizedBox(width: 20),
          Expanded(
            child: Text(
              "Running Dual-Stream CNN, ELA, PRNU, and FFT analyses. Please wait...",
              style: TextStyle(color: Colors.grey),
            ),
          )
        ],
      ),
    );
  }

  Widget _buildVerdictCard(bool isAi, bool isUncertain, ThemeData theme) {
    Color cardColor;
    String titleText;
    IconData icon;

    String rawPred = _predictionResult?['prediction']?.toString().toLowerCase() ?? '';
    if (rawPred.contains('spliced')) {
      cardColor = const Color(0xFFEF4444);
      titleText = "LOCALIZED SPLICING / AI DETECTED";
      icon = Icons.warning_amber_rounded;
    } else if (isAi) {
      cardColor = const Color(0xFFEF4444);
      titleText = "SYNTHETIC / AI DETECTED";
      icon = Icons.warning_amber_rounded;
    } else if (isUncertain) {
      cardColor = const Color(0xFFF59E0B);
      titleText = "INCONCLUSIVE / UNCERTAIN";
      icon = Icons.help_outline_rounded;
    } else {
      cardColor = const Color(0xFF10B981);
      titleText = "VERIFIED ORIGINAL";
      icon = Icons.verified_outlined;
    }

    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: const Color(0xFF161E2E),
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: cardColor.withOpacity(0.3), width: 1.5),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Row(
            children: [
              Icon(icon, color: cardColor, size: 36),
              const SizedBox(width: 14),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      titleText,
                      style: GoogleFonts.outfit(
                        fontWeight: FontWeight.bold,
                        fontSize: 20,
                        color: cardColor,
                      ),
                    ),
                    const SizedBox(height: 2),
                    Text(
                      _predictionResult!['verdict_reason'] ?? 'Analysis complete.',
                      style: TextStyle(color: Colors.grey.shade400, fontSize: 13),
                    ),
                  ],
                ),
              ),
            ],
          ),
          const Divider(height: 28, color: Colors.white12),
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text(
                isAi
                    ? "AI Probability Score"
                    : (isUncertain ? "Forensic Uncertainty Score" : "Authenticity Confidence"),
                style: TextStyle(color: Colors.grey.shade300, fontSize: 14),
              ),
              Text(
                "${_predictionResult!['confidence']}%",
                style: GoogleFonts.outfit(
                  fontWeight: FontWeight.bold,
                  fontSize: 24,
                  color: cardColor,
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildBreakdownCard(ThemeData theme) {
    var breakdown = _predictionResult!['breakdown'];
    if (breakdown == null) return const SizedBox.shrink();

    double cnnScore = breakdown['cnn_score'] ?? 0.0;
    double prnuScore = breakdown['prnu_score'] ?? 0.0;
    double elaScore = breakdown['ela_score'] ?? 0.0;
    double fftScore = breakdown['fft_score'] ?? 0.0;
    double metadataScore = breakdown['metadata_score'] ?? 0.0;

    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: const Color(0xFF161E2E),
        borderRadius: BorderRadius.circular(20),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            "Sub-Detector Analytics",
            style: GoogleFonts.outfit(
              fontWeight: FontWeight.bold,
              fontSize: 18,
            ),
          ),
          const SizedBox(height: 4),
          const Text(
            "Values indicate probability of AI generation",
            style: TextStyle(color: Colors.grey, fontSize: 12),
          ),
          const SizedBox(height: 16),
          _buildMetricProgress("Dual-Stream CNN", cnnScore, const Color(0xFF6366F1)),
          _buildMetricProgress("PRNU Sensor Noise", prnuScore, const Color(0xFF3B82F6)),
          _buildMetricProgress("ELA Error Level", elaScore, const Color(0xFFEC4899)),
          _buildMetricProgress("FFT Frequency Grid", fftScore, const Color(0xFFF59E0B)),
          _buildMetricProgress("EXIF Metadata Check", metadataScore, const Color(0xFF10B981)),
        ],
      ),
    );
  }

  Widget _buildMetricProgress(String name, double val, Color color) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 14),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text(
                name,
                style: const TextStyle(fontWeight: FontWeight.w600, fontSize: 13),
              ),
              Text(
                "${(val * 100).toStringAsFixed(1)}%",
                style: TextStyle(fontWeight: FontWeight.bold, color: color, fontSize: 13),
              ),
            ],
          ),
          const SizedBox(height: 6),
          ClipRRect(
            borderRadius: BorderRadius.circular(4),
            child: LinearProgressIndicator(
              value: val,
              backgroundColor: Colors.white.withOpacity(0.06),
              color: color,
              minHeight: 6,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildMetadataCard(ThemeData theme) {
    var details = _predictionResult!['details'];
    if (details == null) return const SizedBox.shrink();

    String? software = details['software'];
    String? make = details['camera_make'];
    String? model = details['camera_model'];

    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: const Color(0xFF161E2E),
        borderRadius: BorderRadius.circular(20),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Text(
            "EXIF Data Summary",
            style: GoogleFonts.outfit(
              fontWeight: FontWeight.bold,
              fontSize: 18,
            ),
          ),
          const SizedBox(height: 16),
          _buildMetadataRow("Software Signature", software ?? "No signature found"),
          _buildMetadataRow("Camera Manufacturer", make ?? "Unknown"),
          _buildMetadataRow("Camera Model", model ?? "Unknown"),
          if (_predictionResult!['splicing'] != null)
            _buildMetadataRow(
              "Localized Inpainting",
              _predictionResult!['splicing']['is_spliced'] == true
                  ? "SPLICED (${_predictionResult!['splicing']['spliced_area_percentage']}% area, ${_predictionResult!['splicing']['spliced_regions_count']} bounding box)"
                  : "Clean (No localized splicing detected)",
            ),
        ],
      ),
    );
  }

  Widget _buildMetadataRow(String title, String value) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 12),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            title,
            style: TextStyle(color: Colors.grey.shade500, fontSize: 12),
          ),
          const SizedBox(height: 2),
          Text(
            value,
            style: const TextStyle(
              fontWeight: FontWeight.w600,
              fontSize: 14,
              color: Colors.white70,
            ),
          ),
        ],
      ),
    );
  }
}
