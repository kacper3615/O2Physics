// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copdyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file mftMchMatcher.cxx
/// \brief MFT-MCH matching tool for data preparation

#include <TFile.h>
#include <TTree.h>
#include <TKey.h>
#include <TDirectoryFile.h>
#include <TGeoGlobalMagField.h>

#include <DetectorsBase/Propagator.h>
#include <Field/MagneticField.h>
#include <MCHTracking/TrackExtrap.h>
#include <ReconstructionDataFormats/TrackFwd.h>

#include <Math/MatrixRepresentationsStatic.h>
#include <Math/SMatrix.h>
#include <Math/SVector.h>

#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include <cmath>
#include <getopt.h>

using namespace std;

using SMatrix55 = ROOT::Math::SMatrix<double, 5, 5, ROOT::Math::MatRepSym<double, 5>>;
using SMatrix55Std = ROOT::Math::SMatrix<double, 5>;
using SMatrix5 = ROOT::Math::SVector<Double_t, 5>;

// Track parameter structure at matching plane
struct TrackAtPlane {
  float x, y, phi, tanl, invQPt;
  float cXX, cXY, cYY;
  float cPhiX, cPhiY, cPhiPhi;
  float cTglX, cTglY, cTglPhi, cTglTgl;
  float c1PtX, c1PtY, c1PtPhi, c1PtTgl, c1Pt21Pt2;
};

// MFT Track structure
struct MFTTrack {
  int collisionId;
  float z, x, y, phi, tgl, signed1Pt;
  float cXX, cXY, cYY, cPhiX, cPhiY, cPhiPhi;
  float cTglX, cTglY, cTglPhi, cTglTgl;
  float c1PtX, c1PtY, c1PtPhi, c1PtTgl, c1Pt21Pt2;
};

// MCH Track structure
struct MCHTrack {
  int collisionId;
  float z, x, y, phi, tgl, signed1Pt;
  float cXX, cXY, cYY, cPhiX, cPhiY, cPhiPhi;
  float cTglX, cTglY, cTglPhi, cTglTgl;
  float c1PtX, c1PtY, c1PtPhi, c1PtTgl, c1Pt21Pt2;
  
  // Original matching data
  float chi2MatchOriginal;
  int indexMFTOriginal;
};

/// MFT track propagation to Z plane (with or without magnetic field)
template <typename MFT>
TrackAtPlane propagateMFTToZPlane(MFT const& track, float z, bool useMagneticField, o2::field::MagneticField* fieldB)
{
  TrackAtPlane result;

  // Create TrackParCovFwd from MFT track
  SMatrix5 pars(track.x, track.y, track.phi, track.tgl, track.signed1Pt);
  SMatrix55 covs;
  covs(0, 0) = track.cXX;
  covs(0, 1) = track.cXY;
  covs(1, 1) = track.cYY;
  covs(0, 2) = track.cPhiX;
  covs(1, 2) = track.cPhiY;
  covs(2, 2) = track.cPhiPhi;
  covs(0, 3) = track.cTglX;
  covs(1, 3) = track.cTglY;
  covs(2, 3) = track.cTglPhi;
  covs(3, 3) = track.cTglTgl;
  covs(0, 4) = track.c1PtX;
  covs(1, 4) = track.c1PtY;
  covs(2, 4) = track.c1PtPhi;
  covs(3, 4) = track.c1PtTgl;
  covs(4, 4) = track.c1Pt21Pt2;

  o2::track::TrackParCovFwd fwdTrack{track.z, pars, covs, 0.0};

  // Propagate with or without magnetic field
  // When Bz=0, propagateToZ automatically uses linear propagation
  float Bz = 0.0f;
  if (useMagneticField && fieldB) {
    double centerPos[3] = {track.x, track.y, (track.z + z) / 2.0};
    Bz = fieldB->getBz(centerPos);
  }

  fwdTrack.propagateToZ(z, Bz);

  result.x = fwdTrack.getX();
  result.y = fwdTrack.getY();
  result.phi = fwdTrack.getPhi();
  result.tanl = fwdTrack.getTanl();
  result.invQPt = fwdTrack.getInvQPt();

  const auto& cov = fwdTrack.getCovariances();
  result.cXX = cov(0, 0);
  result.cXY = cov(0, 1);
  result.cYY = cov(1, 1);
  result.cPhiX = cov(0, 2);
  result.cPhiY = cov(1, 2);
  result.cPhiPhi = cov(2, 2);
  result.cTglX = cov(0, 3);
  result.cTglY = cov(1, 3);
  result.cTglPhi = cov(2, 3);
  result.cTglTgl = cov(3, 3);
  result.c1PtX = cov(0, 4);
  result.c1PtY = cov(1, 4);
  result.c1PtPhi = cov(2, 4);
  result.c1PtTgl = cov(3, 4);
  result.c1Pt21Pt2 = cov(4, 4);

  return result;
}

/// MCH track propagation to Z plane (with or without magnetic field and absorber)
template <typename MCH>
TrackAtPlane propagateMCHToZPlane(MCH const& track, float z, bool useMagneticField)
{
  TrackAtPlane result;

  Double_t pars[5] = {track.x, track.y, track.phi, track.tgl, track.signed1Pt};
  Double_t covs[15];
  
  int idx = 0;
  covs[idx++] = track.cXX;
  covs[idx++] = track.cXY;
  covs[idx++] = track.cYY;
  covs[idx++] = track.cPhiX;
  covs[idx++] = track.cPhiY;
  covs[idx++] = track.cPhiPhi;
  covs[idx++] = track.cTglX;
  covs[idx++] = track.cTglY;
  covs[idx++] = track.cTglPhi;
  covs[idx++] = track.cTglTgl;
  covs[idx++] = track.c1PtX;
  covs[idx++] = track.c1PtY;
  covs[idx++] = track.c1PtPhi;
  covs[idx++] = track.c1PtTgl;
  covs[idx++] = track.c1Pt21Pt2;

  o2::mch::TrackParam mchTrack(track.z, pars, covs);

  if (useMagneticField) {
    /// Special handling for tracks crossing the absorber with magnetic field
    const float absBack = -505.f;
    const float absFront = -90.f;

    if (track.z < absBack && z > absFront) {
      if (!o2::mch::TrackExtrap::extrapToVertexWithoutBranson(mchTrack, z)) {
        cerr << "Warning: MCH track extrapolation to vertex failed" << endl;
      }
    } else {
      o2::mch::TrackExtrap::extrapToZ(mchTrack, z);
    }
  } else {
    /// Linear propagation without magnetic field
    /// TrackExtrap with no field initialized will use linear propagation
    o2::mch::TrackExtrap::extrapToZ(mchTrack, z);
  }

  result.x = mchTrack.getX();
  result.y = mchTrack.getY();
  result.phi = mchTrack.getPhi();
  result.tanl = mchTrack.getTanl();
  result.invQPt = mchTrack.getInverseBendingMomentum();

  const auto& cov = mchTrack.getCovariances();
  result.cXX = cov(0, 0);
  result.cXY = cov(0, 1);
  result.cYY = cov(1, 1);
  result.cPhiX = cov(0, 2);
  result.cPhiY = cov(1, 2);
  result.cPhiPhi = cov(2, 2);
  result.cTglX = cov(0, 3);
  result.cTglY = cov(1, 3);
  result.cTglPhi = cov(2, 3);
  result.cTglTgl = cov(3, 3);
  result.c1PtX = cov(0, 4);
  result.c1PtY = cov(1, 4);
  result.c1PtPhi = cov(2, 4);
  result.c1PtTgl = cov(3, 4);
  result.c1Pt21Pt2 = cov(4, 4);

  return result;
}

/// Chi2 calculation using covariance-weighted differences in 5D parameter space
float computeMatchChi2(const TrackAtPlane& mch, const TrackAtPlane& mft)
{
  float dx = mch.x - mft.x;
  float dy = mch.y - mft.y;
  float dphi = mch.phi - mft.phi;
  float dtanl = mch.tanl - mft.tanl;
  float dinvqpt = mch.invQPt - mft.invQPt;

  float chi2 = (dx * dx / (mch.cXX + mft.cXX + 1e-10)) +
               (dy * dy / (mch.cYY + mft.cYY + 1e-10)) +
               (dphi * dphi / (mch.cPhiPhi + mft.cPhiPhi + 1e-10)) +
               (dtanl * dtanl / (mch.cTglTgl + mft.cTglTgl + 1e-10)) +
               (dinvqpt * dinvqpt / (mch.c1Pt21Pt2 + mft.c1Pt21Pt2 + 1e-10));

  return chi2;
}

void printUsage(const char* progName)
{
  cout << "Usage: " << progName << " [options]\n"
       << "Options:\n"
       << "  -i, --input <file>       Input AO2D.root file (required)\n"
       << "  -o, --output <file>      Output ROOT file (default: mftMchMatches.root)\n"
       << "  -z, --matching-z <value> Z position of matching plane in cm (default: -77.5)\n"
       << "  -n, --max-candidates <N> Max candidates per MCH track, 0=all (default: 0)\n"
       << "  -f, --use-field          Use magnetic field for propagation (default: linear)\n"
       << "  -h, --help               Show this help message\n"
       << endl;
}

int main(int argc, char** argv)
{
  string inputFile = "";
  string outputFile = "mftMchMatches.root";
  float matchingPlaneZ = -77.5f;
  int maxCandidates = 0;
  bool useField = false;

  static struct option long_options[] = {
    {"input",          required_argument, 0, 'i'},
    {"output",         required_argument, 0, 'o'},
    {"matching-z",     required_argument, 0, 'z'},
    {"max-candidates", required_argument, 0, 'n'},
    {"use-field",      no_argument,       0, 'f'},
    {"help",           no_argument,       0, 'h'},
    {0, 0, 0, 0}
  };

  int opt;
  int option_index = 0;
  while ((opt = getopt_long(argc, argv, "i:o:z:n:f:h", long_options, &option_index)) != -1) {
    switch (opt) {
      case 'i': inputFile = optarg; break;
      case 'o': outputFile = optarg; break;
      case 'z': matchingPlaneZ = atof(optarg); break;
      case 'n': maxCandidates = atoi(optarg); break;
      case 'f': useField = true; break;
      case 'h': printUsage(argv[0]); return 0;
      default: printUsage(argv[0]); return 1;
    }
  }

  if (inputFile.empty()) {
    cerr << "Error: Input file is required\n" << endl;
    printUsage(argv[0]);
    return 1;
  }

  cout << "=== MFT-MCH Matcher ===" << endl;
  cout << "Input file: " << inputFile << endl;
  cout << "Output file: " << outputFile << endl;
  cout << "Matching plane Z: " << matchingPlaneZ << " cm" << endl;
  cout << "Max candidates per track: " << (maxCandidates == 0 ? "all" : to_string(maxCandidates)) << endl;
  cout << "Propagation method: " << (useField ? "Field-based" : "Linear") << endl;
  cout << endl;

  TFile* inFile = TFile::Open(inputFile.c_str(), "READ");
  if (!inFile || inFile->IsZombie()) {
    cerr << "Error: Cannot open input file: " << inputFile << endl;
    return 1;
  }

  o2::field::MagneticField* fieldB = nullptr;
  
  if (useField) {
    cout << "Initializing magnetic field for field-based propagation..." << endl;
    
    try {
      o2::base::Propagator::initFieldFromGRP("GLO/Config/GRPMagField");
      fieldB = static_cast<o2::field::MagneticField*>(TGeoGlobalMagField::Instance()->GetField());
      
      if (fieldB) {
        cout << "Magnetic field initialized successfully from CCDB" << endl;
        o2::mch::TrackExtrap::setField();
      }
    } catch (const std::exception& e) {
      cerr << "Warning: Failed to initialize magnetic field from CCDB: " << e.what() << endl;
      fieldB = nullptr;
    }
    
    if (!fieldB) {
      cerr << "Warning: Could not load magnetic field from CCDB" << endl;
      cerr << "Field-based propagation requires magnetic field data" << endl;
      cerr << "Please use linear propagation instead (omit -f flag)" << endl;
      cerr << "Or set up CCDB access with proper configuration" << endl;
      inFile->Close();
      return 1;
    }
  }

  /// Find DF_* directories containing track data
  vector<string> dfDirs;
  TList* keys = inFile->GetListOfKeys();
  for (int i = 0; i < keys->GetEntries(); i++) {
    TKey* key = (TKey*)keys->At(i);
    string keyName = key->GetName();
    if (keyName.substr(0, 3) == "DF_" && string(key->GetClassName()) == "TDirectoryFile") {
      dfDirs.push_back(keyName);
    }
  }

  if (dfDirs.empty()) {
    cerr << "Error: No DF_* directories found!" << endl;
    inFile->Close();
    return 1;
  }

  cout << "Found " << dfDirs.size() << " DF directories" << endl;

  vector<MFTTrack> mftTracks;
  
  for (const auto& dfDir : dfDirs) {
    TDirectoryFile* df = (TDirectoryFile*)inFile->Get(dfDir.c_str());
    if (!df) continue;
    
    TTree* mftTree = (TTree*)df->Get("O2mfttrack_001");
    if (!mftTree) continue;
    
    MFTTrack track;
    mftTree->SetBranchAddress("fX", &track.x);
    mftTree->SetBranchAddress("fY", &track.y);
    mftTree->SetBranchAddress("fZ", &track.z);
    mftTree->SetBranchAddress("fPhi", &track.phi);
    mftTree->SetBranchAddress("fTgl", &track.tgl);
    mftTree->SetBranchAddress("fSigned1Pt", &track.signed1Pt);
    mftTree->SetBranchAddress("fIndexCollisions", &track.collisionId);
    
    TTree* covTree = (TTree*)df->Get("O2mfttrackcov");
    float sigmaX = 0, sigmaY = 0, sigmaPhi = 0, sigmaTgl = 0, sigma1Pt = 0;
    char rhoXY = 0, rhoPhiX = 0, rhoPhiY = 0, rhoTglX = 0, rhoTglY = 0, rhoTglPhi = 0;
    char rho1PtX = 0, rho1PtY = 0, rho1PtPhi = 0, rho1PtTgl = 0;
    int indexMFTTrack = -1;
    
    bool hasCovTree = (covTree != nullptr);
    map<int, Long64_t> localCovMap;
    
    if (hasCovTree) {
      covTree->SetBranchAddress("fIndexMFTTracks", &indexMFTTrack);
      covTree->SetBranchAddress("fSigmaX", &sigmaX);
      covTree->SetBranchAddress("fSigmaY", &sigmaY);
      covTree->SetBranchAddress("fSigmaPhi", &sigmaPhi);
      covTree->SetBranchAddress("fSigmaTgl", &sigmaTgl);
      covTree->SetBranchAddress("fSigma1Pt", &sigma1Pt);
      covTree->SetBranchAddress("fRhoXY", &rhoXY);
      covTree->SetBranchAddress("fRhoPhiX", &rhoPhiX);
      covTree->SetBranchAddress("fRhoPhiY", &rhoPhiY);
      covTree->SetBranchAddress("fRhoTglX", &rhoTglX);
      covTree->SetBranchAddress("fRhoTglY", &rhoTglY);
      covTree->SetBranchAddress("fRhoTglPhi", &rhoTglPhi);
      covTree->SetBranchAddress("fRho1PtX", &rho1PtX);
      covTree->SetBranchAddress("fRho1PtY", &rho1PtY);
      covTree->SetBranchAddress("fRho1PtPhi", &rho1PtPhi);
      covTree->SetBranchAddress("fRho1PtTgl", &rho1PtTgl);
      
      /// Build index map: fIndexMFTTracks -> covTree entry (sparse mapping)
      Long64_t nCov = covTree->GetEntries();
      for (Long64_t iCov = 0; iCov < nCov; ++iCov) {
        covTree->GetEntry(iCov);
        localCovMap[indexMFTTrack] = iCov;
      }
    }
    
    Long64_t nMFT = mftTree->GetEntries();
    
    for (Long64_t i = 0; i < nMFT; i++) {
      mftTree->GetEntry(i);
      
      sigmaX = sigmaY = sigmaPhi = sigmaTgl = sigma1Pt = 0.0f;
      rhoXY = rhoPhiX = rhoPhiY = rhoTglX = rhoTglY = rhoTglPhi = 0;
      rho1PtX = rho1PtY = rho1PtPhi = rho1PtTgl = 0;
      
      bool hasCov = false;
      if (hasCovTree) {
        int localIndex = static_cast<int>(i);
        auto covIt = localCovMap.find(localIndex);
        if (covIt != localCovMap.end()) {
          covTree->GetEntry(covIt->second);
          hasCov = true;
          
          /// Convert rho (signed char -128..127) to correlation coefficient (-1..1)
          auto rhoToCorr = [](signed char rho) {
            float c = static_cast<float>(rho) / 128.0f;
            return max(-1.0f, min(1.0f, c));
          };
          
          const float kMinSigma = 1e-6f;
          sigmaX = max(sigmaX, kMinSigma);
          sigmaY = max(sigmaY, kMinSigma);
          sigmaPhi = max(sigmaPhi, kMinSigma);
          sigmaTgl = max(sigmaTgl, kMinSigma);
          sigma1Pt = max(sigma1Pt, kMinSigma);
          
          /// Reconstruct covariance matrix: cov[i,j] = rho[i,j] * sigma[i] * sigma[j]
          track.cXX = sigmaX * sigmaX;
          track.cXY = rhoToCorr(rhoXY) * sigmaX * sigmaY;
          track.cYY = sigmaY * sigmaY;
          track.cPhiX = rhoToCorr(rhoPhiX) * sigmaPhi * sigmaX;
          track.cPhiY = rhoToCorr(rhoPhiY) * sigmaPhi * sigmaY;
          track.cPhiPhi = sigmaPhi * sigmaPhi;
          track.cTglX = rhoToCorr(rhoTglX) * sigmaTgl * sigmaX;
          track.cTglY = rhoToCorr(rhoTglY) * sigmaTgl * sigmaY;
          track.cTglPhi = rhoToCorr(rhoTglPhi) * sigmaTgl * sigmaPhi;
          track.cTglTgl = sigmaTgl * sigmaTgl;
          track.c1PtX = rhoToCorr(rho1PtX) * sigma1Pt * sigmaX;
          track.c1PtY = rhoToCorr(rho1PtY) * sigma1Pt * sigmaY;
          track.c1PtPhi = rhoToCorr(rho1PtPhi) * sigma1Pt * sigmaPhi;
          track.c1PtTgl = rhoToCorr(rho1PtTgl) * sigma1Pt * sigmaTgl;
          track.c1Pt21Pt2 = sigma1Pt * sigma1Pt;
        }
      }
      
      if (!hasCov) {
        /// Fallback: no covariance available (track not matched with muon)
        const float kDefaultSigma = 1.0f;
        track.cXX = kDefaultSigma * kDefaultSigma;
        track.cYY = kDefaultSigma * kDefaultSigma;
        track.cPhiPhi = kDefaultSigma * kDefaultSigma;
        track.cTglTgl = kDefaultSigma * kDefaultSigma;
        track.c1Pt21Pt2 = kDefaultSigma * kDefaultSigma;
        track.cXY = 0.f;
        track.cPhiX = 0.f;
        track.cPhiY = 0.f;
        track.cTglX = 0.f;
        track.cTglY = 0.f;
        track.cTglPhi = 0.f;
        track.c1PtX = 0.f;
        track.c1PtY = 0.f;
        track.c1PtPhi = 0.f;
        track.c1PtTgl = 0.f;
      }
      
      mftTracks.push_back(track);
    }
  }

  cout << "Total MFT tracks: " << mftTracks.size() << endl;

  if (mftTracks.empty()) {
    cerr << "Error: No MFT tracks found!" << endl;
    inFile->Close();
    return 1;
  }

  // Create output file and tree
  TFile* outFile = new TFile(outputFile.c_str(), "RECREATE");
  TTree* outTree = new TTree("MFTMCHMatches", "MFT-MCH Matching Results");

  // Output variables
  float out_X_MCH, out_Y_MCH, out_Phi_MCH, out_TanL_MCH, out_InvQPt_MCH;
  float out_X_MFT, out_Y_MFT, out_Phi_MFT, out_TanL_MFT, out_InvQPt_MFT;
  float out_Chi2Match, out_Chi2MatchOriginal;
  int out_Label;

  outTree->Branch("X_MCH", &out_X_MCH);
  outTree->Branch("Y_MCH", &out_Y_MCH);
  outTree->Branch("phi_MCH", &out_Phi_MCH);
  outTree->Branch("tanL_MCH", &out_TanL_MCH);
  outTree->Branch("invqpt_MCH", &out_InvQPt_MCH);
  outTree->Branch("X_MFT", &out_X_MFT);
  outTree->Branch("Y_MFT", &out_Y_MFT);
  outTree->Branch("phi_MFT", &out_Phi_MFT);
  outTree->Branch("tanL_MFT", &out_TanL_MFT);
  outTree->Branch("invqpt_MFT", &out_InvQPt_MFT);
  outTree->Branch("chi2Match", &out_Chi2Match);
  outTree->Branch("chi2MatchOriginal", &out_Chi2MatchOriginal);
  outTree->Branch("label", &out_Label);

  // Statistics
  float minChi2 = 1e10f;
  float maxChi2 = -1e10f;
  double sumChi2 = 0.0;
  int nValidChi2 = 0;
  float minChi2Original = 1e10f;
  float maxChi2Original = -1e10f;
  double sumChi2Original = 0.0;
  int nValidChi2Original = 0;
  int nMatches = 0;

  /// Process MCH tracks with covariances from O2fwdtrack and O2fwdtrackcov
  cout << "Processing MCH tracks and matching..." << endl;
  int totalMchTracks = 0;
  
  bool hasChi2Match = false;
  bool hasMFTIndex = false;

  for (const auto& dfDir : dfDirs) {
    TDirectoryFile* df = (TDirectoryFile*)inFile->Get(dfDir.c_str());
    if (!df) continue;
    
    TTree* mchTree = (TTree*)df->Get("O2fwdtrack");
    if (!mchTree) continue;
    
    MCHTrack mchTrack;
    unsigned char trackType;
    
    mchTree->SetBranchAddress("fX", &mchTrack.x);
    mchTree->SetBranchAddress("fY", &mchTrack.y);
    mchTree->SetBranchAddress("fZ", &mchTrack.z);
    mchTree->SetBranchAddress("fPhi", &mchTrack.phi);
    mchTree->SetBranchAddress("fTgl", &mchTrack.tgl);
    mchTree->SetBranchAddress("fSigned1Pt", &mchTrack.signed1Pt);
    mchTree->SetBranchAddress("fTrackType", &trackType);
    mchTree->SetBranchAddress("fIndexCollisions", &mchTrack.collisionId);
    
    if (dfDir == dfDirs[0]) {
      hasChi2Match = (mchTree->GetBranch("fChi2MatchMCHMFT") != nullptr);
      hasMFTIndex = (mchTree->GetBranch("fIndexMFTTracks") != nullptr);
    }
    
    if (hasChi2Match) {
      mchTree->SetBranchAddress("fChi2MatchMCHMFT", &mchTrack.chi2MatchOriginal);
    }
    if (hasMFTIndex) {
      mchTree->SetBranchAddress("fIndexMFTTracks", &mchTrack.indexMFTOriginal);
    }
    
    /// O2fwdtrackcov: same size as O2fwdtrack, direct 1:1 indexing
    TTree* covTree = (TTree*)df->Get("O2fwdtrackcov");
    float sigmaX = 0, sigmaY = 0, sigmaPhi = 0, sigmaTgl = 0, sigma1Pt = 0;
    char rhoXY = 0, rhoPhiX = 0, rhoPhiY = 0, rhoTglX = 0, rhoTglY = 0, rhoTglPhi = 0;
    char rho1PtX = 0, rho1PtY = 0, rho1PtPhi = 0, rho1PtTgl = 0;
    
    bool hasCovTree = (covTree != nullptr);
    
    if (hasCovTree) {
      covTree->SetBranchAddress("fSigmaX", &sigmaX);
      covTree->SetBranchAddress("fSigmaY", &sigmaY);
      covTree->SetBranchAddress("fSigmaPhi", &sigmaPhi);
      covTree->SetBranchAddress("fSigmaTgl", &sigmaTgl);
      covTree->SetBranchAddress("fSigma1Pt", &sigma1Pt);
      covTree->SetBranchAddress("fRhoXY", &rhoXY);
      covTree->SetBranchAddress("fRhoPhiX", &rhoPhiX);
      covTree->SetBranchAddress("fRhoPhiY", &rhoPhiY);
      covTree->SetBranchAddress("fRhoTglX", &rhoTglX);
      covTree->SetBranchAddress("fRhoTglY", &rhoTglY);
      covTree->SetBranchAddress("fRhoTglPhi", &rhoTglPhi);
      covTree->SetBranchAddress("fRho1PtX", &rho1PtX);
      covTree->SetBranchAddress("fRho1PtY", &rho1PtY);
      covTree->SetBranchAddress("fRho1PtPhi", &rho1PtPhi);
      covTree->SetBranchAddress("fRho1PtTgl", &rho1PtTgl);
    }
    
    Long64_t nMCH = mchTree->GetEntries();
    
    for (Long64_t i = 0; i < nMCH; i++) {
      mchTree->GetEntry(i);
      
      /// Only process standalone MCH tracks (trackType == 0)
      if (trackType != 0) continue;
      
      totalMchTracks++;
      if (totalMchTracks % 100 == 0) {
        cout << "Processed " << totalMchTracks << " MCH tracks..." << "\r" << flush;
      }

      if (hasCovTree) {
        /// Direct indexing: O2fwdtrackcov[i] corresponds to O2fwdtrack[i]
        covTree->GetEntry(i);
        
        auto rhoToCorr = [](signed char rho) {
          float c = static_cast<float>(rho) / 128.0f;
          return max(-1.0f, min(1.0f, c));
        };
        
        const float kMinSigma = 1e-6f;
        sigmaX = max(sigmaX, kMinSigma);
        sigmaY = max(sigmaY, kMinSigma);
        sigmaPhi = max(sigmaPhi, kMinSigma);
        sigmaTgl = max(sigmaTgl, kMinSigma);
        sigma1Pt = max(sigma1Pt, kMinSigma);
        
        mchTrack.cXX = sigmaX * sigmaX;
        mchTrack.cXY = rhoToCorr(rhoXY) * sigmaX * sigmaY;
        mchTrack.cYY = sigmaY * sigmaY;
        mchTrack.cPhiX = rhoToCorr(rhoPhiX) * sigmaPhi * sigmaX;
        mchTrack.cPhiY = rhoToCorr(rhoPhiY) * sigmaPhi * sigmaY;
        mchTrack.cPhiPhi = sigmaPhi * sigmaPhi;
        mchTrack.cTglX = rhoToCorr(rhoTglX) * sigmaTgl * sigmaX;
        mchTrack.cTglY = rhoToCorr(rhoTglY) * sigmaTgl * sigmaY;
        mchTrack.cTglPhi = rhoToCorr(rhoTglPhi) * sigmaTgl * sigmaPhi;
        mchTrack.cTglTgl = sigmaTgl * sigmaTgl;
        mchTrack.c1PtX = rhoToCorr(rho1PtX) * sigma1Pt * sigmaX;
        mchTrack.c1PtY = rhoToCorr(rho1PtY) * sigma1Pt * sigmaY;
        mchTrack.c1PtPhi = rhoToCorr(rho1PtPhi) * sigma1Pt * sigmaPhi;
        mchTrack.c1PtTgl = rhoToCorr(rho1PtTgl) * sigma1Pt * sigmaTgl;
        mchTrack.c1Pt21Pt2 = sigma1Pt * sigma1Pt;
      } else {
        const float kDefaultSigma = 1.0f;
        mchTrack.cXX = kDefaultSigma * kDefaultSigma;
        mchTrack.cYY = kDefaultSigma * kDefaultSigma;
        mchTrack.cPhiPhi = kDefaultSigma * kDefaultSigma;
        mchTrack.cTglTgl = kDefaultSigma * kDefaultSigma;
        mchTrack.c1Pt21Pt2 = kDefaultSigma * kDefaultSigma;
        mchTrack.cXY = 0.f;
        mchTrack.cPhiX = 0.f;
        mchTrack.cPhiY = 0.f;
        mchTrack.cTglX = 0.f;
        mchTrack.cTglY = 0.f;
        mchTrack.cTglPhi = 0.f;
        mchTrack.c1PtX = 0.f;
        mchTrack.c1PtY = 0.f;
        mchTrack.c1PtPhi = 0.f;
        mchTrack.c1PtTgl = 0.f;
      }
      
      if (!hasChi2Match) {
        mchTrack.chi2MatchOriginal = -1.0f;
      }
      if (!hasMFTIndex) {
        mchTrack.indexMFTOriginal = -1;
      }

      int collisionId = mchTrack.collisionId;
      
      /// Propagate MCH track to matching plane
      TrackAtPlane mchAtPlane = propagateMCHToZPlane(mchTrack, matchingPlaneZ, useField);

      /// Find matching MFT tracks from same collision
      struct Candidate {
        size_t idx;
        TrackAtPlane mftAtPlane;
        float chi2;
      };
      vector<Candidate> candidates;

      for (size_t i = 0; i < mftTracks.size(); i++) {
        if (mftTracks[i].collisionId != collisionId) continue;

        TrackAtPlane mftAtPlane = propagateMFTToZPlane(mftTracks[i], matchingPlaneZ, useField, fieldB);

        float chi2 = computeMatchChi2(mchAtPlane, mftAtPlane);
        candidates.push_back({i, mftAtPlane, chi2});
      }

      if (candidates.empty()) continue;

      sort(candidates.begin(), candidates.end(),
           [](const Candidate& a, const Candidate& b) { return a.chi2 < b.chi2; });

      int nToSave = maxCandidates;
      if (nToSave == 0 || nToSave > (int)candidates.size()) {
        nToSave = candidates.size();
      }

      float chi2Original = mchTrack.chi2MatchOriginal;

      for (int iCand = 0; iCand < nToSave; iCand++) {
        out_X_MCH = mchAtPlane.x;
        out_Y_MCH = mchAtPlane.y;
        out_Phi_MCH = mchAtPlane.phi;
        out_TanL_MCH = mchAtPlane.tanl;
        out_InvQPt_MCH = mchAtPlane.invQPt;

        out_X_MFT = candidates[iCand].mftAtPlane.x;
        out_Y_MFT = candidates[iCand].mftAtPlane.y;
        out_Phi_MFT = candidates[iCand].mftAtPlane.phi;
        out_TanL_MFT = candidates[iCand].mftAtPlane.tanl;
        out_InvQPt_MFT = candidates[iCand].mftAtPlane.invQPt;

        out_Chi2Match = candidates[iCand].chi2;
        out_Chi2MatchOriginal = chi2Original;
        out_Label = (iCand == 0) ? 1 : 0;

        if (isfinite(out_Chi2Match) && out_Chi2Match >= 0.0f) {
          minChi2 = min(minChi2, out_Chi2Match);
          maxChi2 = max(maxChi2, out_Chi2Match);
          sumChi2 += out_Chi2Match;
          nValidChi2++;
        }

        if (isfinite(out_Chi2MatchOriginal) && out_Chi2MatchOriginal >= 0.0f) {
          minChi2Original = min(minChi2Original, out_Chi2MatchOriginal);
          maxChi2Original = max(maxChi2Original, out_Chi2MatchOriginal);
          sumChi2Original += out_Chi2MatchOriginal;
          nValidChi2Original++;
        }

        outTree->Fill();
        nMatches++;
      }
    }
  }

  /// DEBUG statistics
  cout << endl;
  cout << "\n=== Results ===" << endl;
  cout << "Total MFT tracks: " << mftTracks.size() << endl;
  cout << "Total MCH tracks processed: " << totalMchTracks << endl;
  cout << "Total matches saved: " << nMatches << endl;
  
  if (nValidChi2 > 0) {
    cout << "\nChi2 Match Statistics:" << endl;
    cout << "  Valid entries: " << nValidChi2 << endl;
    cout << "  Min: " << minChi2 << endl;
    cout << "  Max: " << maxChi2 << endl;
    cout << "  Avg: " << (sumChi2 / nValidChi2) << endl;
  }

  if (nValidChi2Original > 0) {
    cout << "\nChi2 Match Original Statistics:" << endl;
    cout << "  Valid entries: " << nValidChi2Original << endl;
    cout << "  Min: " << minChi2Original << endl;
    cout << "  Max: " << maxChi2Original << endl;
    cout << "  Avg: " << (sumChi2Original / nValidChi2Original) << endl;
  } else {
    cout << "\nChi2 Match Original: Not available (no fChi2MatchMCHMFT branch in input file)" << endl;
  }

  outFile->cd();
  outTree->Write();
  outFile->Close();
  inFile->Close();

  cout << "\nOutput written to: " << outputFile << endl;
  cout << "Done!" << endl;

  return 0;
}
