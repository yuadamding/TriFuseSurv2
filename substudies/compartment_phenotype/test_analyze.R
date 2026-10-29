#!/usr/bin/env Rscript
# Numerical/guard fixtures only. No project patient table is read.
file_arg <- grep('^--file=', commandArgs(FALSE),value=TRUE)
HERE <- dirname(normalizePath(sub('^--file=','',file_arg[1])))
source(file.path(HERE,'analyze.R'))

run_tests <- function() {
  checks <- 0L
  check <- function(value,label) {
    if(!isTRUE(value)) stop('FAILED: ',label,call.=FALSE)
    checks <<- checks+1L; cat('PASS:',label,'\n')
  }
  rejects <- function(expression,pattern=NULL) {
    error <- tryCatch({force(expression);NULL},error=function(e)e)
    inherits(error,'error') && (is.null(pattern)||grepl(pattern,conditionMessage(error),ignore.case=TRUE))
  }
  p <- yaml::read_yaml(file.path(HERE,'protocol.yaml'))
  set.seed(82713); n <- 420L
  d <- data.frame(patient_id=sprintf('synthetic-%03d',seq_len(n)),
    split=rep(c('train','val'),length.out=n),AGE=runif(n,35,80),SEX=sample(c('M','F'),n,TRUE),
    T=sample(c('1','2','3','4','4B'),n,TRUE),N=sample(c('0','1','2A','2B','2C','3'),n,TRUE),
    HPV=sample(c('0','1','NA','nd'),n,TRUE),M=runif(n,.05,.2),D=runif(n,-.02,.02),
    VP_cm3=exp(rnorm(n,2,.6)),VN_cm3=exp(rnorm(n,1,.8)),
    phiP=runif(n,.25,.9),entry_days=NA_real_,timing_status='unknown')
  d$HP <- d$M-d$D; d$HN <- d$M+d$D
  d$nP <- round(d$VP_cm3*4000); d$nN <- round(d$VN_cm3*4000)
  d$f0P <- d$f1P <- d$f0N <- d$f1N <- 0
  d$original_z_spacing_mm <- 1
  event <- rexp(n,1/4000); censor <- rexp(n,1/6500)
  d$OS_TIME <- pmin(event,censor); d$OS_EVENT <- as.integer(event<=censor)
  d$analysis_row_id <- d$patient_id
  trans <- processing(d); x <- transform_data(d,trans)
  primary <- primary_pair(x)
  check(primary$usable && length(coef(primary$full$fit))==6 && length(coef(primary$reduced$fit))==5,
        'Primary nested models have exactly five and six slopes')
  stage <- x; stage$Tfactor<-factor(stage$T);stage$Nfactor<-factor(stage$N)
  extended <- fit_pair(stage,c(primary_terms,'Tfactor','Nfactor'),'D',15)
  check(extended$usable && length(coef(extended$full$fit))==15,
        'Five T and six N nominal categories add nine parameters, full fifteen slopes')
  temporal <- temporal_pair(x,p$models$temporal_change_point_days)
  check(temporal$usable && length(coef(temporal$reduced$fit))==7 && length(coef(temporal$full$fit))==8,
        'Temporal common-M reduced/full contrast models have seven/eight slopes')
  check(all(c('Mlate','Dlate') %in% names(coef(temporal$full$fit))) &&
        'Mlate' %in% names(coef(temporal$reduced$fit)), 'Temporal contrast test adjusts common M time change in both models')
  effects <- compartment_effects(primary,trans$SD_D)
  check(abs(effects$nodal_per_normalized_intensity_unit$coefficient-
            effects$primary_per_normalized_intensity_unit$coefficient-
            effects$difference_per_normalized_intensity_unit$coefficient) < 1e-12,
        'Transformed nodal-minus-primary coefficient equals direct beta_D')
  b <- coef(primary$full$fit); v <- vcov(primary$full$fit)
  expected <- (v['M','M']+v['D','D']-2*v['M','D'])/4
  check(abs(effects$primary_per_normalized_intensity_unit$SE^2-expected)<1e-12,
        'Compartment confidence interval uses full M/D covariance')
  check(abs(effects$contrast_HR_per_fixed_development_SD_D$log_HR-b['D']*trans$SD_D)<1e-12,
        'Contrast HR is scaled by one frozen development SD_D')
  direct <- fit_checked(as.formula(paste(survival_lhs(x),'~ AGE10+male+logVP+logVN+HP+HN')),x,6)
  check(direct$usable && abs(direct$fit$loglik[2]-primary$full$fit$loglik[2])<1e-8,
        'M/D and unstandardized HP/HN parameterizations give equal likelihood')
  check(abs(coef(direct$fit)['HP']-effects$primary_per_normalized_intensity_unit$coefficient)<1e-8,
        'Primary physical-scale coefficient agrees with direct HP/HN fit')
  check(qr(model.matrix(~HP+HN+D,data=x))$rank < ncol(model.matrix(~HP+HN+D,data=x)),
        'Adding D to both HP and HN creates detected exact collinearity')
  damaged <- x; damaged$D[1] <- NA_real_
  bad <- primary_pair(damaged)
  check(!bad$usable, 'No valid nested likelihood comparison when added-term missingness differs')
  cut <- p$models$temporal_change_point_days
  edge <- x[1:4,,drop=FALSE]; edge$OS_TIME<-c(cut,cut+1,cut-1,cut+100);edge$OS_EVENT<-c(1,1,0,1)
  expanded <- temporal_expand(edge,cut)
  check(sum(expanded$event)==sum(edge$OS_EVENT) && all(expanded$start<expanded$stop),
        'Start/stop construction counts each death once and preserves valid intervals')
  check(sum(expanded$patient_id==edge$patient_id[1])==1 &&
        expanded$late[expanded$patient_id==edge$patient_id[1]]==0,
        'A death at exactly day1826 belongs to the early interval')
  delayed <- edge; delayed$entry_days<-c(10,cut+.5,0,cut+10)
  delayed_expanded <- temporal_expand(delayed,cut)
  check(min(delayed_expanded$start[delayed_expanded$patient_id==delayed$patient_id[2]])==cut+.5,
        'Authentic late entry does not create an earlier risk interval')
  check(time_mode(x)=='exploratory_retrospective_diagnosis_OS_timing_unverified' && all(is.na(x$entry_days)),
        'Unknown clinical timing stays unknown; retrospective analysis is not verified baseline prediction')
  mixed <- x; mixed$entry_days[1]<-0
  check(rejects(time_mode(mixed),'mixed'),
        'Mixed authentic and unknown entry cannot silently discard known delayed entry')
  invalid_entry <- x; invalid_entry$entry_days<-0; invalid_entry$entry_days[1]<-invalid_entry$OS_TIME[1]
  check(rejects(time_mode(invalid_entry),'precede'), 'Event or censoring at entry is not a valid risk interval')
  invalid_entry$entry_days[1]<-invalid_entry$OS_TIME[1]+1
  check(rejects(time_mode(invalid_entry),'precede'), 'Event or censoring before entry is rejected')
  invalid_entry$entry_days[1]<-Inf
  check(rejects(time_mode(invalid_entry),'finite'), 'Nonfinite authentic entry cannot become unknown retrospective entry')
  known <- x[1:3,,drop=FALSE];known$entry_days<-c(0,cut,cut+100)
  known$OS_TIME<-known$entry_days+c(1000,2000,3000);known$OS_EVENT<-c(1,0,1)
  calibration <- calibrate_design(known)
  check(calibration$observed_duration_median==2000 && calibration$calibration_time_scale=='since_authentic_entry',
        'Authentic-entry simulation calibration uses observed post-entry duration')
  retrospective_calibration <- calibrate_design(x)
  check(retrospective_calibration$observed_duration_median==median(x$OS_TIME) &&
        retrospective_calibration$calibration_time_scale=='supplied_OS_duration_retrospective_timing_unknown',
        'All-unknown retrospective design retains the supplied OS-duration calibration')
  mock_absolute_cut <- function() {
    env<-environment(simulate_times);had<-exists('rexp',envir=env,inherits=FALSE)
    if(had) saved<-get('rexp',envir=env,inherits=FALSE)
    on.exit(if(had)assign('rexp',saved,envir=env) else rm('rexp',envir=env),add=TRUE)
    calls<-0L
    assign('rexp',function(n,rate) {
      calls<<-calls+1L
      rep(switch(calls,5000,100,10000),n)
    },envir=env)
    simulate_times(known,p,trans,'common_M_temporal_stress_D_zero')
  }
  absolute_cut <- mock_absolute_cut()
  check(identical(absolute_cut$data$OS_TIME,c(cut+100,cut+100,cut+200)),
        'Temporal stress changes at absolute diagnosis day1826 for early, at-cut and already-late entries')
  absolute_intervals <- temporal_expand(absolute_cut$data,cut)
  check(sum(absolute_intervals$event)==3 &&
        min(absolute_intervals$start[absolute_intervals$patient_id==known$patient_id[3]])==cut+100,
        'Entry-aware temporal simulation/expansion preserves one death per patient and no pre-entry rows')
  check(rejects(finite_sd(rep(0,12))),'Zero contrast variation is unestimable without arbitrary denominator')
  sampled <- resample_patients(x,c(4,4,2,7))
  check(!anyDuplicated(sampled$patient_id) && identical(sampled$HP,x$HP[c(4,4,2,7)]) &&
        identical(sampled$HN,x$HN[c(4,4,2,7)]), 'Patient bootstrap preserves both compartments within each sampled row')
  set.seed(74); sim <- simulate_times(x,p,trans,'common_M_temporal_stress_D_zero')
  check(sim$true_standardized_D==0 && identical(sim$data$HP,x$HP) && identical(sim$data$HN,x$HN) &&
        any(sim$data$OS_TIME!=x$OS_TIME), 'Design simulation preserves covariates and generates synthetic outcomes only')
  fixture_protocol <- p; fixture_protocol$simulation$minimum_usable_fraction<-0
  fixture_protocol$simulation$maximum_null_rejection_fraction<-1
  fixture_protocol$simulation$CI_coverage_range<-c(0,1)
  fixture_protocol$simulation$maximum_median_log_HR_CI_width<-0
  # Nine tiny synthetic design replicates exercise gate routing, not a real
  # design assessment or a replacement for the frozen 1,000/scenario inventory.
  fixture_design <- design_assessment(x,fixture_protocol,trans,simulations=3)
  check(!fixture_design$scenarios$null_D$checks$median_interval_width &&
        !fixture_design$scenarios$alternative_D_HR1_5$checks$median_interval_width,
        'Primary null and alternative scenarios enforce prespecified width precision gate')
  stress <- fixture_design$scenarios$common_M_temporal_stress_D_zero
  check(!stress$interval_width_is_gating && !('median_interval_width' %in% names(stress$checks)) &&
        is.finite(stress$median_log_HR_CI_width),
        'Temporal Dlate width is reported separately and cannot veto primary D precision')
  shape <- fit_pair(x,c('AGE10','male','logVP'),'phiP',4)
  check(shape$usable && length(coef(shape$full$fit))==4, 'Fallback has four slopes, no accidental sixth parameter')
  fixture <- tempfile('phenotype-statistics-fixtures-');dir.create(fixture)
  on.exit(unlink(fixture,recursive=TRUE),add=TRUE)
  dir.create(file.path(fixture,'qualification'))
  qpath <- file.path(fixture,'qualification','paired_decision.json')
  write_json(qpath,list(passed=FALSE,protocol_sha256='fixture'))
  check(rejects(require_qualification(fixture,'fixture'),'failed'), 'Failed measurement qualification forbids design/association stage')
  check(rejects(require_qualification(fixture,'other-protocol'),'different protocol'), 'Qualification hash mismatch fails closed')
  fallback_allowed(fixture)
  dir.create(file.path(fixture,'statistics'))
  write_json(file.path(fixture,'statistics','PRIMARY_ASSOCIATION_STARTED.json'),list(fixture=TRUE))
  check(rejects(fallback_allowed(fixture),'after'), 'Fallback cannot be activated after real paired association inspection sentinel')
  unlink(file.path(fixture,'statistics','PRIMARY_ASSOCIATION_STARTED.json'))
  write_json(file.path(fixture,'statistics','primary_result.json'),list(p_value=1))
  check(rejects(fallback_allowed(fixture),'after'), 'A null paired result cannot trigger fallback promotion')
  cohort_fixture <- file.path(fixture,'cohort.csv');feature_fixture<-file.path(fixture,'features.csv')
  writeLines('synthetic-cohort',cohort_fixture);writeLines('synthetic-feature',feature_fixture)
  unlink(qpath);write_json(qpath,list(passed=TRUE,protocol_sha256='fixture',
      cohort_manifest_sha256=sha(cohort_fixture),baseline_feature_file_sha256=sha(feature_fixture)))
  bound_paths <- list(cohort_manifest_csv=cohort_fixture,phenotype_features_csv=feature_fixture)
  accepted <- require_qualification(fixture,'fixture',paths=bound_paths)
  check(accepted$decision$passed,'Passing qualification binds exact cohort and baseline feature files')
  cat('changed',file=feature_fixture,append=TRUE)
  check(rejects(require_qualification(fixture,'fixture',paths=bound_paths),'changed'),
        'Feature mutation after perturbation qualification forbids later analysis')
  shape_fixture <- file.path(fixture,'primary_only_features.csv');writeLines('synthetic-shape',shape_fixture)
  fallback_qpath <- file.path(fixture,'qualification','primary_only_decision.json')
  fallback_q <- list(passed=TRUE,protocol_sha256='fixture',feature_name='phiP',
      cohort_manifest_sha256=sha(cohort_fixture),baseline_feature_file_sha256=sha(shape_fixture),
      supporting_measurement_file_sha256=sha(feature_fixture))
  write_json(fallback_qpath,fallback_q)
  check(require_qualification(fixture,'fixture',fallback=TRUE,paths=bound_paths)$decision$passed,
        'Fallback qualification binds sphericity and supporting intensity/voxel-volume tables separately')
  unlink(fallback_qpath);bad_fallback_q<-fallback_q
  bad_fallback_q$supporting_measurement_file_sha256<-paste(rep('0',64),collapse='')
  write_json(fallback_qpath,bad_fallback_q)
  check(rejects(require_qualification(fixture,'fixture',fallback=TRUE,paths=bound_paths),'supporting'),
        'A valid sphericity hash cannot conceal invalid fallback supporting measurement binding')
  unlink(fallback_qpath);bad_fallback_q<-fallback_q
  bad_fallback_q$baseline_feature_file_sha256<-paste(rep('0',64),collapse='')
  write_json(fallback_qpath,bad_fallback_q)
  check(rejects(require_qualification(fixture,'fixture',fallback=TRUE,paths=bound_paths),'changed'),
        'A valid supporting measurement hash cannot conceal invalid fallback sphericity binding')
  budget_marker <- file.path(fixture,'statistics','FALLBACK_DESIGN_STARTED.json')
  write_json(budget_marker,list(synthetic_fixture=TRUE))
  check(rejects(write_json(budget_marker,list(synthetic_fixture=TRUE)),'exclusive'),
        'An exclusive fallback design START marker prevents a second admitted simulation inventory')
  protocol_copy <- file.path(fixture,'protocol.yaml')
  protocol_fixture <- p;protocol_fixture$paths$run_root<-fixture
  yaml::write_yaml(protocol_fixture,protocol_copy)
  dir.create(file.path(fixture,'protocol'))
  lock <- list(protocol_sha256=sha(protocol_copy),source_commit='fixture',code_sha256=setNames(list(sha(SCRIPT_FILE)),'analyze.R'))
  write_json(file.path(fixture,'protocol','PROTOCOL_LOCK.json'),lock)
  verified <- verify_protocol(protocol_copy,fixture)
  check(verified$protocol_sha256==sha(protocol_copy), 'Frozen protocol and exact statistical source hash accepted')
  wrong_root <- file.path(fixture,'wrong-root');dir.create(wrong_root)
  check(rejects(verify_protocol(protocol_copy,wrong_root),'run root'),
        'Canonical actual run root must equal the frozen protocol run root')
  cat('\n# fixture mutation\n',file=protocol_copy,append=TRUE)
  check(rejects(verify_protocol(protocol_copy,fixture),'hash'), 'Protocol mutation after freeze is rejected')
  cat('\nAll',checks,'statistical fixture checks passed. No real cohort or phenotype-OS association was read.\n')
}
run_tests()
