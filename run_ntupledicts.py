import os
from sys import argv
from uproot import open as uproot_open
from ntupledicts import operations as ndops
from ntupledicts.operations import select as sel
from ntupledicts import plot as ndplot
from ntupledicts.ml import data as ndmldata
from ntupledicts.ml import predict as ndmlpred
from ntupledicts.ml import models as ndmlmodels
from ntupledicts.ml import plot as ndmlplot
from matplotlib.pyplot import cla, sca, gca, savefig
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Softmax
from tensorflow.keras.layers import Dense



# I/O global variables
# input_files = ["eventsets/ZMM_PU200_D49.root",
#     "eventsets/ZEE_PU200_D49.root",
#     "eventsets/QCD_PU200_D49.root"]
input_files = ["eventsets/TTbar_PU200_D49_20.root"]
input_file_short = "mlrepsample"
output_dir = "/Users/caseypancoast/Desktop/plotbbs/"


def main(argv):

    # Open ntuples, specify desired properties and cuts to be applied
    event_sets = []
    for input_file in input_files:
        event_sets.append(next(iter(uproot_open(input_file).values()))["eventTree"])
    properties_by_track_type = {"trk": ["pt", "eta", "z0", "nstub", "chi2", "bendchi2", "genuine", "matchtp_pdgid"],
                                "matchtrk": ["pt", "eta", "nstub", "chi2", "bendchi2"],
                                "tp": ["pt", "eta", "nstub", "dxy", "d0", "eventid", "nmatch"]}
    general_cut_dicts = {"tp": {"eta": sel(-2.4, 2.4), "pt": sel(2, 100), "nstub": sel(4, 9999),
                        "dxy": sel(-1.0, 1.0), "d0": sel(-1.0, 1.0), "eventid": sel(0)}}

    # Create ntuple properties dict from event set, apply cuts
    ntuple_dict = ndops.uproot_ntuples_to_ntuple_dict(event_sets, properties_by_track_type)
    # ntuple_dict = ndops.cut_ntuple_dict(ntuple_dict, {"trk": {"pt": sel(20, 100)}})
    all_nd_gens = ndops.cut_ntuple_dict(ntuple_dict, {"trk": {"genuine": sel(1)}})
    nd_fakes = ndops.cut_ntuple_dict(ntuple_dict, {"trk": {"genuine": sel(0)}})
    nd_gens = ndops.reduce_ntuple_dict(all_nd_gens, track_limit=ndops.track_prop_dict_length(nd_fakes["trk"]), shuffle_tracks=True, seed=42)
    nd_both = ndops.shuffle_ntuple_dict(ndops.add_ntuple_dicts([nd_gens, nd_fakes]), seed=42)
    # nd_both = ndops.reduce_ntuple_dict(nd_both, track_limit=1000)
    go(nd_both)

    print("Process complete. Exiting program.")


def go(ntuple_dict):

    # Make datasets
    train_ds, eval_ds, test_ds = ndmldata.TrackPropertiesDataset(ntuple_dict["trk"],
            "genuine", ["chi2", "bendchi2", "nstub"]).split([.7, .2, .1])

    # Train models on dataset
    NN = ndmlmodels.make_neuralnet(train_ds, eval_dataset=eval_ds, hidden_layers=[15, 8], epochs=10)
    GBDT = ndmlmodels.make_gbdt(train_ds)
    cuts = [{"chi2rphi": sel(0, 23), "chi2rz": sel(0, 7), "chi2": sel(0, 21)}]

    test_ds.add_prediction("NN", ndmlpred.predict_labels(NN, test_ds.get_data()))
    test_ds.add_prediction("GBDT", ndmlpred.predict_labels(GBDT, test_ds.get_data()))
    test_ds.add_prediction("cuts", ndmlpred.predict_labels_cuts(next(iter(cuts)), test_ds))

    test(test_ds)


def test(test_ds):
    """Use this function when testing samples or functionality rather
    than actually running stuff."""

    # TODO okay, labels work. Now, are TPR/FPR calculated accurately?

    # Tensorflow labels and predictions
    actual_labels = test_ds.get_labels()
    model_pred_labels = test_ds.get_prediction("GBDT")
    thresh_mpl = ndmlpred.apply_threshhold(model_pred_labels, .1)
    cut_pred_labels = test_ds.get_prediction("cuts")

    for labels in zip(actual_labels, model_pred_labels, thresh_mpl):
        print(labels)

    print(ndmlpred.false_positive_rate(actual_labels, model_pred_labels))
    print(ndmlpred.false_positive_rate(actual_labels, thresh_mpl))


def plot(test_ds):
    """Make Claire's plots."""

    # ROC curve
    ax = ndmlplot.plot_rocs(test_ds, [NN, GBDT], ["NN", "GBDT"], cuts)
    sca(ax)
    savefig(output_dir + "roc_curve.pdf")
    cla()

    # TPR/FPR vs some track prop for NN, GBDT, cuts
    bins=11  # for pt
    binning_prop="pt"
    for pred_comparison, pred_comp_name in zip(
            [ndmlpred.true_positive_rate, ndmlpred.false_positive_rate],
            ["TPR", "FPR"]):
        ax = gca()
        for pred_name in test_ds.get_all_prediction_names():
            ax = ndmlplot.plot_pred_comparison_by_track_property(
                    test_ds, pred_name, pred_comparison,
                    binning_prop, bins=bins, legend_id=pred_name, ax=ax)
        sca(ax)
        ax.set_ylabel(pred_comp_name)
        ax.set_title("{} by {}".format(pred_comp_name, binning_prop))
        ax.legend()
        savefig("{}{}_vs_{}_by_model".format(output_dir, pred_comp_name, binning_prop))
        cla()


    # selectors using pdgid
    el_sel = sel([sel(11), sel(-11)])
    muon_sel = sel([sel(13), sel(-13)])
    fake_sel = sel(-999)  # ...right?
    hadron_sel = sel([el_sel, muon_sel, fake_sel], invert=True)  # Everything else, at least
                                                                 # for these samples

    # TPR/FPR vs some track prop * models, ov. particle type
    for pred_comparison, pred_comp_name in zip(
            [ndmlpred.true_positive_rate, ndmlpred.false_positive_rate],
            ["TPR", "FPR"]):
        for pred_name in test_ds.get_all_prediction_names():
            ax = gca()
            for pdgid_sel, particle_type in zip(
                    [el_sel, muon_sel, hadron_sel, fake_sel],
                    ["electrons", "muons", "hadrons", "fakes"]):
                pdgid_sel_dict = {"matchtp_pdgid": pdgid_sel}
                test_ds_pdgid = test_ds.cut(pdgid_sel_dict)
                ax = ndmlplot.plot_pred_comparison_by_track_property(
                        test_ds_pdgid, pred_name,
                        pred_comparison, binning_prop, bins=bins,
                        legend_id=particle_type, ax=ax)
            ax.set_ylabel(pred_comp_name)
            ax.set_title("{} of {} by pdgid".format(pred_comp_name, pred_name))
            ax.legend()
            savefig("{}{}_{}_by_pdgid.pdf".format(output_dir, pred_comp_name, pred_name))
            cla()

    # TPR/FPR vs decision thresh * models, ov. particle type
    for pred_comparison, pred_comp_name in zip(
            [ndmlpred.true_positive_rate, ndmlpred.false_positive_rate],
            ["TPR", "FPR"]):
        for pred_name in test_ds.get_all_prediction_names():
            ax = gca()
            for pdgid_sel, particle_type in zip(
                    [el_sel, muon_sel, hadron_sel, fake_sel],
                    ["electrons", "muons", "hadrons", "fakes"]):
                pdgid_sel_dict = {"matchtp_pdgid": pdgid_sel}
                test_ds_pdgid = test_ds.cut(pdgid_sel_dict)
                ax = ndmlplot.plot_pred_comparison_by_threshhold(test_ds_pdgid,
                        pred_name, pred_comparison, legend_id=particle_type, ax=ax)
            ax.set_ylabel(pred_comp_name)
            ax.set_title("{} of {} by pdgid".format(pred_comp_name, pred_name))
            ax.legend()
            savefig("{}{}_{}_by_pdgid.pdf".format(output_dir, pred_comp_name, pred_name))
            cla()


main(argv)

