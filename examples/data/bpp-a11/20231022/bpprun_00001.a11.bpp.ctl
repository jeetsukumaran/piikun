
          seed =  -1

       seqfile = bpprun_00001.input.chars.txt
      Imapfile = bpprun_00001.input.imap.txt
       outfile = bpprun_00001.a11.results.out.txt
      mcmcfile = bpprun_00001.a11.results.mcmc.txt

* speciesdelimitation = 0 * fixed species tree
* speciesdelimitation = 1 0 2    * species delimitation rjMCMC algorithm0 and finetune(e)
  speciesdelimitation = 1 1 2 1 * species delimitation rjMCMC algorithm1 finetune (a m)
          speciestree = 1        * species tree NNI/SPR
*        speciestree = 1  0.4 0.2 0.1   * speciestree pSlider ExpandRatio ShrinkRatio


     speciesmodelprior = 1         * 0: uniform labeled histories; 1:uniform rooted trees

  species&tree = 6  S1.sub1 S4.sub1 S3.sub1 S6.sub1 S5.sub1 S2.sub1
                     4 4 4 4 4 4
                 ((S1.sub1,(S4.sub1,(S3.sub1,(S6.sub1,S5.sub1)))),S2.sub1);


       usedata = 1    * 0: no data (prior); 1:seq like
         nloci = 10    * number of data sets in seqfile

     cleandata = 0    * remove sites with ambiguity data (1:yes, 0:no)?

    *  - Root age = 17.373977840963924; N = 1000000.0; mu = 1e-08;
    *  - Optimized Priors:
    *      - thetaprior = 3.0 0.03454470536566532 * Mean = 0.01727235268283266
    *      - tauprior = 12.0 1.6280000000000003e-05 * Mean = 0.014800000000000004;
    *  - Default Priors:
    *      - thetaprior = 3.0 0.002
    *      - tauprior = 3.0 0.03
    thetaprior = 3.0 0.03454470536566532
      tauprior = 12.0   1.6280000000000003e-05

      finetune =  1: 3 0.003 0.002 0.00002 0.005 0.9 0.001 0.001 # finetune for GBtj, GBspr, theta, tau, mix

         print = 1 0 0 0   * MCMC samples, locusrate, heredityscalars Genetrees
        burnin = 4000
      sampfreq = 100
       nsample = 1000

